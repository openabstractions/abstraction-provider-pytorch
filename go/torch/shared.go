package torch

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/openabstractions/abstraction-resource/go/holder"
)

// torchShared is the one OA PyTorch engine process the PyTorch kinds share:
// a video set, an image set, or both resident in one process, as MuseTree's
// engine holds FLUX.2 [klein] and Hummingbird-XT (PYTORCH-T10).
//
// Each set is its own holder component with its own lease. The set that
// starts the process holds its own measured lease, and its process is
// bound to that lease; stopping that set stops the process. A set that
// joins a running process holds the difference between the two sets'
// measured lease together and the first set's own, so the two leases sum
// to what both measured in one process. Unloading the joined set unloads
// it in the engine and gives its lease back, which leaves the first set's
// own. resource/leases@1 grants a fixed amount and has no resize (RES-L1,
// RES-L2), so the process's reservation grows and shrinks by this second
// lease rather than by changing the first.
//
// Every engine operation holds the process's one operation slot: a load
// (the start of the process or a set joining it), an unload and a job. An
// unload therefore waits for the running job instead of queueing behind it
// in the engine, and while any operation holds the slot every set reads
// busy, so neither the idle time nor a yield takes a set from under it. The
// lock order is a set's holder component lock, then the slot: a job holds
// its set's use and loads it before it holds the slot, and a load or an
// unload takes the slot inside the component lock the holder holds. A
// driver reset stops the process without the slot.
type torchShared struct {
	host   *Host
	engine TorchEngine
	// active reports a set's queued or running work, by set name.
	active map[string]func() bool

	// starting serializes process starts. It is held while a start proves
	// and holds files, which can take minutes on a first launch; mu is held
	// only briefly, because the holder reads busy under its component locks
	// while it renews every lease.
	starting sync.Mutex

	mu      sync.Mutex
	running *torchProcess
	// base is the set whose lease is the running process's own.
	base string
	// last is the latest process started, kept after it exits so a refused
	// load can still be reported.
	last *torchProcess

	// slot admits one engine operation at a time: a load, an unload or a
	// job (hold).
	slot chan struct{}

	// beforeStart, set by tests only, runs once a start holds starting.
	beforeStart func()
	// unloadWithin replaces stopWithin for an unload's answer; only tests
	// set it.
	unloadWithin time.Duration
	// beforeUnload, set by tests only, runs in a joined set's Stop once it
	// holds its operation, before the unload is sent.
	beforeUnload func()
}

func newTorchShared(h *Host, engine TorchEngine) *torchShared {
	return &torchShared{host: h, engine: engine, active: map[string]func() bool{}, slot: make(chan struct{}, 1)}
}

// busy reports whether name's holder component is in use: while any
// engine operation holds the process every set is, and otherwise its own
// queued or running work and, for the set whose lease the process is bound
// to, the work of every set in the same process, since stopping it stops
// them all.
func (s *torchShared) busy(name string) bool {
	if len(s.slot) > 0 {
		return true
	}
	if f := s.active[name]; f != nil && f() {
		return true
	}
	s.mu.Lock()
	base := s.base
	s.mu.Unlock()
	if base != name {
		return false
	}
	for other, f := range s.active {
		if other != name && f != nil && f() {
			return true
		}
	}
	return false
}

// serves reports whether name is a set of this engine.
func (s *torchShared) serves(name string) bool {
	_, ok := s.engine.place(name)
	return ok
}

// leaseFor is the amount name's component asks for beside a running
// process whose own set is base: what both measured together less base's
// own lease, or name's own lease when the two were never measured together.
func (s *torchShared) leaseFor(sp TorchSetPlace, base string) int64 {
	other, ok := s.engine.place(base)
	if !ok {
		return sp.Set.Lease
	}
	together, _, ok := sp.Set.beside(other.Set, other.Options, sp.Options)
	if !ok || together-other.Set.Lease < 1 {
		return sp.Set.Lease
	}
	return together - other.Set.Lease
}

// ensure loads name in the shared engine under its lease and returns the
// running process (PYTORCH-T5, T10). A load planned for a process that
// exited before it ran is planned again.
func (s *torchShared) ensure(ctx context.Context, name string) (*torchProcess, error) {
	sp, ok := s.engine.place(name)
	if !ok {
		return nil, refuse(NoEngine, "the PyTorch engine serves no model set %q", name)
	}
	h := s.host
	for attempt := 0; ; attempt++ {
		var amount int64
		err := h.resources.Ensure(ctx, name, func() (holder.Component, error) {
			if err := s.engine.present(ctx, sp); err != nil {
				var refusal *Refusal
				if errors.As(err, &refusal) {
					return holder.Component{}, refusal
				}
				return holder.Component{}, refuse(NoEngine, "%v", err)
			}
			s.mu.Lock()
			running, base := s.running, s.base
			s.mu.Unlock()
			if running == nil || running.exited() || base == name {
				amount = sp.Set.Lease
				return holder.Component{Amount: amount, Start: func() (holder.Owned, error) { return s.startWith(sp) }}, nil
			}
			amount = s.leaseFor(sp, base)
			return holder.Component{Amount: amount, Start: func() (holder.Owned, error) { return s.join(running, sp) }}, nil
		})
		if errors.Is(err, errTorchEngineChanged) && attempt < 2 {
			continue
		}
		if err != nil {
			s.reportLoad(name)
			return nil, h.mapFailure(name, amount, err)
		}
		s.mu.Lock()
		running := s.running
		s.mu.Unlock()
		if running == nil || !running.conn.loaded(name) {
			return nil, errors.New("the PyTorch engine is not running")
		}
		return running, nil
	}
}

// startWith starts the engine process with sp as its first set. A process
// of this engine that still runs is stopped first: its own set's component
// is no longer resident, so nothing holds a lease for it. The start is a
// load: it holds the operation slot until the component's readiness ends.
func (s *torchShared) startWith(sp TorchSetPlace) (holder.Owned, error) {
	s.starting.Lock()
	defer s.starting.Unlock()
	if s.beforeStart != nil {
		s.beforeStart()
	}
	release := s.take()
	s.mu.Lock()
	old, base := s.running, s.base
	s.mu.Unlock()
	if old != nil && !old.exited() {
		if base != sp.Set.Name {
			// Another set started the process since this load was planned.
			release()
			return nil, errTorchEngineChanged
		}
		old.process.stop(stopWithin)
		if !old.exited() {
			release()
			return nil, errors.New("the PyTorch engine's earlier process did not stop")
		}
	}
	running, err := s.engine.start(sp)
	if err != nil {
		release()
		return nil, err
	}
	s.mu.Lock()
	s.running, s.base, s.last = running, sp.Set.Name, running
	s.mu.Unlock()
	go func() {
		<-running.process.exited
		s.mu.Lock()
		if s.running == running {
			s.running, s.base = nil, ""
		}
		s.mu.Unlock()
	}()
	ready := s.engine.Ready
	if ready <= 0 {
		ready = DefaultVideoReady
	}
	return &torchOwned{ownedProcess: ownedProcess{name: sp.Set.Name, process: running.process, ready: ready, host: s.host},
		t: running, set: sp.Set.Name, release: release}, nil
}

// join asks running to load sp beside the set it holds. The join is a load:
// it holds the operation slot until the set is ready, or, when its
// readiness fails, until its Stop has unloaded it.
func (s *torchShared) join(running *torchProcess, sp TorchSetPlace) (holder.Owned, error) {
	release := s.take()
	s.mu.Lock()
	current := s.running
	s.mu.Unlock()
	if current != running || running.exited() {
		release()
		return nil, errTorchEngineChanged
	}
	state, err := running.add(sp)
	if err != nil {
		release()
		return nil, err
	}
	ready := s.engine.Ready
	if ready <= 0 {
		ready = DefaultVideoReady
	}
	return &torchJoined{shared: s, t: running, set: sp.Set.Name, state: state, ready: ready, loading: release}, nil
}

// take waits for the operation slot and returns its release, which runs
// once however often it is called. A load takes it inside its component's
// lock, which the holder holds through Start and readiness: the lock order
// is the component's lock, then the slot.
func (s *torchShared) take() func() {
	s.slot <- struct{}{}
	var once sync.Once
	return func() { once.Do(func() { <-s.slot }) }
}

// torchOwned is the component of the set that started the process: the
// process itself, bound to that set's lease. Its readiness is the process's
// and its first set's, and ends the start's hold of the operation slot.
type torchOwned struct {
	ownedProcess
	t       *torchProcess
	set     string
	release func()
}

func (o *torchOwned) Ready(ctx context.Context) error {
	defer o.release()
	if err := o.t.bootstrap(ctx); err != nil {
		return err
	}
	if err := o.ownedProcess.Ready(ctx); err != nil {
		return err
	}
	if !o.t.conn.loaded(o.set) {
		return errors.New("the PyTorch engine did not load its first model set")
	}
	return nil
}

// Stop stops the process; a stop needs no slot.
func (o *torchOwned) Stop() bool {
	defer o.release()
	return o.ownedProcess.Stop()
}

// torchJoined is the component of a set loaded into a process another set
// started. No process is bound to its lease: one child binds to one lease
// (RES-L2). It lives while the process lives and holds the set.
type torchJoined struct {
	shared *torchShared
	t      *torchProcess
	set    string
	state  *torchSetState
	ready  time.Duration

	// loading releases the join's hold of the operation slot. Ready
	// releases it once the set is ready; after a failed readiness the
	// holder's Stop unloads the set first, still inside the join's
	// operation, and then releases it.
	loading func()
}

func (j *torchJoined) PID() int { return 0 }

// Ready waits for the engine's ready for the set; a refused load is the
// engine's code, and the holder then stops the component and gives its
// lease back.
func (j *torchJoined) Ready(ctx context.Context) error {
	timer := time.NewTimer(j.ready)
	defer timer.Stop()
	select {
	case <-j.state.ready:
		j.loading()
		return nil
	case code := <-j.state.failed:
		// The code stays readable for the caller's report.
		select {
		case j.state.failed <- code:
		default:
		}
		return refuse(NoEngine, "the PyTorch engine refused to load %s with code %s; its log has the reason", j.set, code)
	case <-j.t.process.exited:
		return refuse(NoEngine, "the PyTorch engine exited while loading %s", j.set)
	case <-j.t.conn.closed:
		return refuse(NoEngine, "the PyTorch engine broke its protocol while loading %s", j.set)
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return fmt.Errorf("the PyTorch engine did not load %s within %s", j.set, j.ready)
	}
}

func (j *torchJoined) Alive() bool {
	if j.t.exited() {
		return false
	}
	select {
	case <-j.state.unloaded:
		return false
	default:
		return true
	}
}

// Stop unloads the set in the engine and lets go of its files; only then
// does the holder release its lease. The unload is an engine operation.
// Once the set was ready, Stop takes the operation slot, waiting for the
// operation that holds it, a running job included, or for the process's
// exit. When the set's readiness failed, a cancelled wait included, the
// join's own operation still holds the slot and the engine may still be
// loading the set: Stop sends the unload within that operation. Either way
// it waits up to stopWithin for the engine's unloaded, its refusal of the
// load, or the process's exit, and stops the process otherwise: only that
// is sure to free the set's memory.
func (j *torchJoined) Stop() bool {
	defer j.t.release(j.set)
	ready := false
	select {
	case <-j.state.ready:
		ready = true
	default:
	}
	if ready {
		// The join's operation ended at readiness.
		if !j.Alive() {
			return true
		}
		release, gone := j.shared.holdUntilExit(j.t)
		if gone {
			return true
		}
		defer release()
	} else {
		defer j.loading()
	}
	if !j.Alive() || len(j.state.failed) > 0 {
		// Gone, or the engine refused the load and dropped the set itself.
		return true
	}
	if j.shared.beforeUnload != nil {
		j.shared.beforeUnload()
	}
	if err := j.t.conn.send(map[string]any{"op": "unload", "set": j.set}, nil); err == nil {
		within := j.shared.unloadWithin
		if within <= 0 {
			within = stopWithin
		}
		timer := time.NewTimer(within)
		defer timer.Stop()
		select {
		case <-j.state.unloaded:
			return true
		case code := <-j.state.failed:
			// The engine refused the load and dropped the set; its
			// unloaded follows the refusal. The code stays readable for
			// the caller's report.
			select {
			case j.state.failed <- code:
			default:
			}
			return true
		case <-j.t.process.exited:
			return true
		case <-timer.C:
		}
	}
	j.t.process.stop(stopWithin)
	return j.t.exited()
}

// hold takes the process's operation slot for a job; release lets the next
// operation in. A job holds its set's use (holder Begin) and has its set
// loaded (ensure, whose load takes the slot itself) before it holds the
// slot: the lock order is the component's lock, then the slot.
func (s *torchShared) hold(ctx context.Context) (release func(), err error) {
	select {
	case s.slot <- struct{}{}:
		return func() { <-s.slot }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// acquire loads name under its lease and then holds the operation slot for
// one job on it. A set that left the process between its load and the job's
// hold (the process exited or was replaced) is loaded again.
func (s *torchShared) acquire(ctx context.Context, name string) (*torchProcess, func(), error) {
	for attempt := 0; ; attempt++ {
		running, err := s.ensure(ctx, name)
		if err != nil {
			return nil, nil, err
		}
		release, err := s.hold(ctx)
		if err != nil {
			return nil, nil, err
		}
		if !running.exited() && running.conn.loaded(name) {
			return running, release, nil
		}
		release()
		if attempt >= 2 {
			return nil, nil, errors.New("the PyTorch engine is not running")
		}
	}
}

// holdUntilExit takes the operation slot for an unload of a set of t, or
// reports t gone once its process exits first: a job that ends its engine
// stops the process before it asks the holder to unload.
func (s *torchShared) holdUntilExit(t *torchProcess) (release func(), gone bool) {
	select {
	case s.slot <- struct{}{}:
		return func() { <-s.slot }, false
	case <-t.process.exited:
		return nil, true
	}
}

// lose stops the running process at once, without the operation slot: a
// driver reset waits for no job (RES-G2). Every set's component then reads
// gone, and its Stop needs no slot.
func (s *torchShared) lose() {
	s.mu.Lock()
	running := s.running
	s.mu.Unlock()
	if running != nil {
		running.process.stop(stopWithin)
	}
}

// torchLoser is what the GPU fault handling stops on a driver reset: the
// PyTorch engine's process first, directly, then every component through
// the holder (RES-G2).
type torchLoser struct{ h *Host }

func (l torchLoser) LoseAll(why string) []holder.Lost {
	if l.h.torch != nil {
		l.h.torch.lose()
	}
	return l.h.resources.LoseAll(why)
}

// reportLoad names, through the provider's error channel, the code the
// engine last refused name's load with and the stage it reached; the
// engine's own text stays in its log.
func (s *torchShared) reportLoad(name string) {
	s.mu.Lock()
	running := s.last
	s.mu.Unlock()
	report := s.host.options.OnError
	if running == nil || report == nil {
		return
	}
	state := running.conn.state(name)
	if state == nil {
		return
	}
	select {
	case code := <-state.failed:
		report(fmt.Errorf("pytorch: the PyTorch engine refused to load %s with code %s at stage %q; its log has the reason",
			name, code, running.conn.stageOf(name)))
	default:
	}
}

// reportPeak reports, through the provider's error channel, a job whose
// peak reservation passed what the process's resident sets lease: residency
// above the estimate stays charged (RES-L2).
func (s *torchShared) reportPeak(sp TorchSetPlace, peak int64) {
	report := s.host.options.OnError
	if report == nil {
		return
	}
	limit := sp.Set.Lease
	s.mu.Lock()
	running := s.running
	s.mu.Unlock()
	if running != nil {
		for _, other := range s.engine.sets() {
			if other.Set.Name == sp.Set.Name || !running.conn.loaded(other.Set.Name) {
				continue
			}
			if together, _, ok := sp.Set.beside(other.Set, other.Options, sp.Options); ok {
				limit = together
			} else {
				limit += other.Set.Lease
			}
		}
	}
	if peak > limit {
		report(fmt.Errorf("pytorch: the PyTorch engine reserved %d bytes of device memory, above the %d-byte lease of %s",
			peak, limit, sp.Set.Name))
	}
}

// stop ends running, the only sure end of a job the engine did not finish,
// and gives name's lease back. The process's other set gives its own back
// once the holder sees the process gone.
func (s *torchShared) stop(running *torchProcess, name, why string) {
	running.process.stop(stopWithin)
	s.host.resources.Unload(name, why)
}
