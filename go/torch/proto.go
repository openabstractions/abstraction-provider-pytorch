package torch

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"
)

// The OA PyTorch engine's stdio protocol (CONTRACT.md PYTORCH-T3): one
// header line, a JSON object of at most torchMaxHeader bytes ending in a line
// feed, and, when the header carries "bytes": N, exactly N raw bytes after
// it. The provider writes the engine's stdin and reads its stdout; the
// engine's stderr is its log. Every message the engine sends is checked
// against the closed list below before anything acts on it, and the first
// one that fails ends the engine: its text never reaches a caller.

const (
	torchProtocol  = 2
	torchMaxHeader = 64 << 10
	// torchMaxSetName bounds a model set's name in a message.
	torchMaxSetName = 64
)

// torchCodes are the failure codes an engine's error and failed messages
// may carry.
var torchCodes = []string{"load", "files", "source", "config", "oom", "device", "input", "error"}

// torchMessage is one message the engine sends. Fields outside a type's own
// are refused.
type torchMessage struct {
	Type    string `json:"type"`
	Set     string `json:"set,omitempty"`
	ID      string `json:"id,omitempty"`
	Code    string `json:"code,omitempty"`
	Stage   string `json:"stage,omitempty"`
	Phase   string `json:"phase,omitempty"`
	Done    int    `json:"done,omitempty"`
	Total   int    `json:"total,omitempty"`
	Index   int    `json:"index,omitempty"`
	Frames  int    `json:"frames,omitempty"`
	Peak    int64  `json:"peak,omitempty"`
	Seq     int64  `json:"seq,omitempty"`
	Healthy *bool  `json:"healthy,omitempty"`
	PID     int    `json:"pid,omitempty"`
	Bytes   int64  `json:"bytes,omitempty"`
	// Protocol is the hello's version.
	Protocol int `json:"protocol,omitempty"`

	// payload is the binary frame after the header.
	payload []byte
}

// torchFields are the header keys each message type may carry.
var torchFields = map[string][]string{
	"hello":     {"type", "protocol", "pid"},
	"loading":   {"type", "set", "stage"},
	"ready":     {"type", "set"},
	"error":     {"type", "set", "code"},
	"unloaded":  {"type", "set"},
	"pong":      {"type", "seq", "healthy"},
	"progress":  {"type", "id", "phase", "done", "total"},
	"frame":     {"type", "id", "index", "bytes"},
	"done":      {"type", "id", "frames", "peak"},
	"failed":    {"type", "id", "code"},
	"cancelled": {"type", "id"},
}

// errTorchProtocol wraps every message the provider refuses.
var errTorchProtocol = errors.New("the PyTorch engine broke its protocol")

// readTorchMessage reads one message. frameBytes is the payload a frame
// message must carry, zero when no job expects frames.
func readTorchMessage(r *bufio.Reader, frameBytes int64) (torchMessage, error) {
	line, err := readTorchLine(r)
	if err != nil {
		return torchMessage{}, err
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(line, &keys); err != nil {
		return torchMessage{}, fmt.Errorf("%w: a header is not one JSON object", errTorchProtocol)
	}
	var m torchMessage
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&m); err != nil {
		return torchMessage{}, fmt.Errorf("%w: a header does not decode: %v", errTorchProtocol, err)
	}
	allowed, ok := torchFields[m.Type]
	if !ok {
		return torchMessage{}, fmt.Errorf("%w: message type %q", errTorchProtocol, m.Type)
	}
	for key := range keys {
		if !slices.Contains(allowed, key) {
			return torchMessage{}, fmt.Errorf("%w: a %s message carries %q", errTorchProtocol, m.Type, key)
		}
	}
	switch m.Type {
	case "loading", "ready", "error", "unloaded":
		if m.Set == "" || len(m.Set) > torchMaxSetName {
			return torchMessage{}, fmt.Errorf("%w: a %s message names no model set", errTorchProtocol, m.Type)
		}
	}
	switch m.Type {
	case "error", "failed":
		if !slices.Contains(torchCodes, m.Code) {
			return torchMessage{}, fmt.Errorf("%w: code %q", errTorchProtocol, m.Code)
		}
	case "frame":
		if frameBytes <= 0 || m.Bytes != frameBytes {
			return torchMessage{}, fmt.Errorf("%w: a frame of %d bytes where %d are expected", errTorchProtocol, m.Bytes, frameBytes)
		}
		m.payload = make([]byte, m.Bytes)
		if _, err := io.ReadFull(r, m.payload); err != nil {
			return torchMessage{}, fmt.Errorf("%w: a frame ended early", errTorchProtocol)
		}
	case "pong":
		if m.Healthy == nil {
			return torchMessage{}, fmt.Errorf("%w: a pong without its health", errTorchProtocol)
		}
	case "hello":
		if m.Protocol != torchProtocol || m.PID <= 0 {
			return torchMessage{}, fmt.Errorf("%w: hello speaks protocol %d", errTorchProtocol, m.Protocol)
		}
	case "loading":
		if len(m.Stage) > 64 {
			return torchMessage{}, fmt.Errorf("%w: a loading stage of %d bytes", errTorchProtocol, len(m.Stage))
		}
	case "progress":
		if (m.Phase != "step" && m.Phase != "decode") || m.Total < 1 || m.Total > 10000 || m.Done < 1 || m.Done > m.Total {
			return torchMessage{}, fmt.Errorf("%w: progress %s %d/%d", errTorchProtocol, m.Phase, m.Done, m.Total)
		}
	}
	if m.Type != "frame" && m.Bytes != 0 {
		return torchMessage{}, fmt.Errorf("%w: a %s message carries a payload", errTorchProtocol, m.Type)
	}
	return m, nil
}

// readTorchLine reads one header line of at most torchMaxHeader bytes.
func readTorchLine(r *bufio.Reader) ([]byte, error) {
	var line []byte
	for {
		chunk, err := r.ReadSlice('\n')
		line = append(line, chunk...)
		if len(line) > torchMaxHeader {
			return nil, fmt.Errorf("%w: a header passes %d bytes", errTorchProtocol, torchMaxHeader)
		}
		if err == nil {
			return line, nil
		}
		if !errors.Is(err, bufio.ErrBufferFull) {
			if len(line) > 0 && errors.Is(err, io.EOF) {
				return nil, fmt.Errorf("%w: the output ended inside a header", errTorchProtocol)
			}
			return nil, err
		}
	}
}

// writeTorchMessage writes one header, and payload after it when there is
// one, as the engine reads them.
func writeTorchMessage(w io.Writer, header map[string]any, payload []byte) error {
	if len(payload) > 0 {
		header["bytes"] = len(payload)
	}
	line, err := json.Marshal(header)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	if len(line) > torchMaxHeader {
		return fmt.Errorf("a header of %d bytes passes the engine's bound", len(line))
	}
	if _, err := w.Write(line); err != nil {
		return err
	}
	if len(payload) > 0 {
		if _, err := w.Write(payload); err != nil {
			return err
		}
	}
	return nil
}

// torchConn is the provider's end of one engine's pipes. One goroutine reads
// the engine's stdout; one job at a time receives its messages. A message
// that breaks the protocol, and the output's end, close the connection, and
// broken then reports why. Each model set the provider asked the engine to
// load has its own state; a message about any other set breaks the
// protocol.
type torchConn struct {
	wmu sync.Mutex
	w   io.WriteCloser

	// ready closes when the first set loaded: the process is ready.
	hello  chan struct{}
	ready  chan struct{}
	closed chan struct{}

	mu    sync.Mutex
	err   error
	job   *torchSink
	seq   int64
	pongs map[int64]chan bool
	pid   int
	first string
	sets  map[string]*torchSetState
}

// torchSetState is one model set's load in one engine: ready closes when it
// loaded, failed carries the code its load was refused with, and unloaded
// closes when the engine dropped it.
type torchSetState struct {
	ready    chan struct{}
	failed   chan string
	unloaded chan struct{}
	stage    string
	isReady  bool
	gone     bool
}

// torchSink receives one job's messages. frameBytes is the size of one of
// its frames.
type torchSink struct {
	id         string
	frameBytes int64
	messages   chan torchMessage
	// gone closes when the job stops listening; a message after it is
	// dropped, and the next one for the job breaks the protocol.
	gone chan struct{}
}

// newTorchConn reads the engine at r and writes it at w; first is the set
// its launch loads.
func newTorchConn(r io.Reader, w io.WriteCloser, first string) *torchConn {
	c := &torchConn{w: w, hello: make(chan struct{}), ready: make(chan struct{}), closed: make(chan struct{}),
		pongs: map[int64]chan bool{}, first: first, sets: map[string]*torchSetState{}}
	c.expect(first)
	go c.read(bufio.NewReaderSize(r, 64<<10))
	return c
}

// expect registers a load of set the provider is about to send and returns
// its state; an earlier state of the same set is replaced.
func (c *torchConn) expect(set string) *torchSetState {
	s := &torchSetState{ready: make(chan struct{}), failed: make(chan string, 1), unloaded: make(chan struct{})}
	c.mu.Lock()
	c.sets[set] = s
	c.mu.Unlock()
	return s
}

// state is set's state, or nil when no load of it was asked for.
func (c *torchConn) state(set string) *torchSetState {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sets[set]
}

// loaded reports whether set is loaded in the engine now.
func (c *torchConn) loaded(set string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	s := c.sets[set]
	return s != nil && s.isReady && !s.gone
}

// stageOf is the last loading stage set reported.
func (c *torchConn) stageOf(set string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if s := c.sets[set]; s != nil {
		return s.stage
	}
	return ""
}

// read takes the engine's messages until its output ends or one breaks the
// protocol.
func (c *torchConn) read(r *bufio.Reader) {
	err := c.loop(r)
	c.mu.Lock()
	if c.err == nil {
		c.err = err
	}
	for seq, ch := range c.pongs {
		close(ch)
		delete(c.pongs, seq)
	}
	c.mu.Unlock()
	close(c.closed)
}

func (c *torchConn) loop(r *bufio.Reader) error {
	hello := false
	for {
		c.mu.Lock()
		var expect int64
		if c.job != nil {
			expect = c.job.frameBytes
		}
		c.mu.Unlock()
		m, err := readTorchMessage(r, expect)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return io.EOF
			}
			return err
		}
		if !hello {
			if m.Type != "hello" {
				return fmt.Errorf("%w: the engine began with %s", errTorchProtocol, m.Type)
			}
			hello = true
			c.mu.Lock()
			c.pid = m.PID
			c.mu.Unlock()
			close(c.hello)
			continue
		}
		switch m.Type {
		case "hello":
			return fmt.Errorf("%w: a second hello", errTorchProtocol)
		case "loading", "ready", "error", "unloaded":
			if err := c.setMessage(m); err != nil {
				return err
			}
		case "pong":
			c.mu.Lock()
			ch, ok := c.pongs[m.Seq]
			delete(c.pongs, m.Seq)
			c.mu.Unlock()
			if ok {
				ch <- *m.Healthy
				close(ch)
			}
		default:
			c.mu.Lock()
			job := c.job
			c.mu.Unlock()
			if job == nil || m.ID != job.id {
				return fmt.Errorf("%w: a %s message for a job not running", errTorchProtocol, m.Type)
			}
			select {
			case job.messages <- m:
			case <-job.gone:
			}
		}
	}
}

// setMessage applies one message about a model set's load: loading and
// error before ready, one ready, and unloaded once loaded or refused.
func (c *torchConn) setMessage(m torchMessage) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	s := c.sets[m.Set]
	if s == nil || s.gone {
		return fmt.Errorf("%w: a %s message for a set not being loaded", errTorchProtocol, m.Type)
	}
	switch m.Type {
	case "loading":
		if s.isReady {
			return fmt.Errorf("%w: loading after ready", errTorchProtocol)
		}
		s.stage = m.Stage
	case "ready":
		if s.isReady {
			return fmt.Errorf("%w: a second ready", errTorchProtocol)
		}
		s.isReady = true
		close(s.ready)
		if m.Set == c.first {
			close(c.ready)
		}
	case "error":
		if s.isReady {
			return fmt.Errorf("%w: a load error after ready", errTorchProtocol)
		}
		select {
		case s.failed <- m.Code:
		default:
			return fmt.Errorf("%w: a second load error", errTorchProtocol)
		}
	case "unloaded":
		s.gone = true
		close(s.unloaded)
	}
	return nil
}

// broken is why the connection closed, or nil while it is open.
func (c *torchConn) broken() error {
	select {
	case <-c.closed:
	default:
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

// send writes one message to the engine's stdin.
func (c *torchConn) send(header map[string]any, payload []byte) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	return writeTorchMessage(c.w, header, payload)
}

// begin makes sink the job that receives messages; end forgets it.
func (c *torchConn) begin(id string, frameBytes int64) *torchSink {
	sink := &torchSink{id: id, frameBytes: frameBytes, messages: make(chan torchMessage, 4), gone: make(chan struct{})}
	c.mu.Lock()
	c.job = sink
	c.mu.Unlock()
	return sink
}

func (c *torchConn) end(sink *torchSink) {
	c.mu.Lock()
	if c.job == sink {
		c.job = nil
	}
	c.mu.Unlock()
	close(sink.gone)
}

// readyNow reports whether the engine said it is ready.
func (c *torchConn) readyNow() bool {
	select {
	case <-c.ready:
		return true
	default:
		return false
	}
}

// ping asks the engine whether it still runs and its device answers; an
// engine that does not answer within ctx is not healthy.
func (c *torchConn) ping(ctx context.Context) bool {
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return false
	}
	c.seq++
	seq := c.seq
	ch := make(chan bool, 1)
	c.pongs[seq] = ch
	c.mu.Unlock()
	if err := c.send(map[string]any{"op": "ping", "seq": seq}, nil); err != nil {
		c.mu.Lock()
		delete(c.pongs, seq)
		c.mu.Unlock()
		return false
	}
	select {
	case healthy, ok := <-ch:
		return ok && healthy
	case <-ctx.Done():
		c.mu.Lock()
		delete(c.pongs, seq)
		c.mu.Unlock()
		return false
	case <-c.closed:
		return false
	}
}
