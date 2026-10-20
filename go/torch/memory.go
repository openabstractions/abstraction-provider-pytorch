package torch

import (
	"context"
	"errors"
	"log"
	"sync"
	"time"

	"github.com/openabstractions/abstraction-provider-kit/go/modelbytes"
	"github.com/openabstractions/abstraction-provider-kit/go/modelfiles"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// ModelMemory lends each model file to an engine in its sandbox as a sealed
// memory section OA filled from the held file and proved (provider kit
// INF-MS9, KIT-MB1..KIT-MB4). The engine maps the section through the model
// bytes receiver, copies the tensors to the device and releases it; the
// lease's end frees the section.
type ModelMemory struct {
	// Records proves a later fill of the same file by OA's span record
	// (INF-MS7); nil hashes every fill whole.
	Records *modelfiles.SpanRecords
	// Hold takes the abstraction.resource holding of n bytes of memory the
	// section is charged to, and returns its release, which the kit calls
	// once when the section is freed. nil counts nothing.
	Hold func(n int64) (release func(), err error)

	mu   sync.Mutex
	live int
	idle chan struct{}
}

// pipeWriteAccess is FILE_GENERIC_WRITE, what the engine's copy of a lease
// pipe end may do; the engine only closes it.
const pipeWriteAccess = 0x120116

// heldFile is the held model file as a storage store of one object: the
// located reference a fill takes (INF-MS9).
type heldFile struct {
	path, digest string
	size         int64
}

func (h heldFile) Name() string { return "pytorch-held" }
func (h heldFile) Find(digest string) (storage.Ref, bool) {
	return storage.NewRef(h.Name(), h.digest, h.path, h.size), digest == h.digest
}
func (h heldFile) Place(string, int64) (storage.Ref, error) {
	return storage.Ref{}, storage.ErrReadOnly
}
func (h heldFile) Path(r storage.Ref) string { return storage.Locator(r) }

// pendingLend is one model file a loading set's engine may ask for.
type pendingLend struct {
	file  VideoModelFile
	final string
	// ref is the archive's reference when the archive serves the file.
	ref storage.Ref
}

// await records that set's engine may ask for f, held at final.
func (t *torchProcess) await(set string, f VideoModelFile, final string, ref storage.Ref) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.pending == nil {
		t.pending = map[string]map[string]pendingLend{}
	}
	if t.pending[set] == nil {
		t.pending[set] = map[string]pendingLend{}
	}
	t.pending[set][f.Role] = pendingLend{file: f, final: final, ref: ref}
}

// answerNeed lends the model file of role the engine's load of set asks for,
// once: a section OA fills now, or the file's handle when the section cannot
// be made. A role not awaited, or a second ask, is answered without a URI,
// which fails the load.
func (t *torchProcess) answerNeed(set, role string) {
	t.mu.Lock()
	p, ok := t.pending[set][role]
	delete(t.pending[set], role)
	t.mu.Unlock()
	answer := map[string]any{"op": "lent", "set": set, "role": role}
	if ok && p.ref.Digest != "" {
		if uri, err := t.engine.Memory.lendArchived(t, p.file, p.ref); err == nil {
			answer["uri"] = uri
		} else {
			// The archive's folder never reaches a log line: the error
			// is named by its class alone.
			log.Printf("pytorch: %s could not be filled from the archive: %s", p.file.Role, archiveFailure(err))
		}
	} else if ok {
		if uri, err := t.lendModel(p.file, p.final); err == nil {
			answer["uri"] = uri
		} else {
			log.Printf("pytorch: %s could not be lent to the engine: %v", p.file.Name, err)
		}
	}
	//unchecked: a send that fails means the engine is gone; its exit ends the load
	t.conn.send(answer, nil)
}

// lend fills f's section from the held file final, lends it and a lease
// pipe end to the running engine, and returns the URI the engine maps.
func (m *ModelMemory) lend(t *torchProcess, f VideoModelFile, final string) (string, error) {
	store := heldFile{path: final, digest: "sha256:" + f.SHA256, size: f.Size}
	ref, _ := store.Find(store.digest)
	return m.lendFrom(t, f, store, ref)
}

// lendFrom fills f's section from ref, a located reference store issued,
// and lends it to the running engine.
func (m *ModelMemory) lendFrom(t *torchProcess, f VideoModelFile, store storage.Store, ref storage.Ref) (string, error) {
	if t.process == nil || t.process.owned == nil || t.conn == nil {
		return "", errors.New("no running engine to lend to")
	}
	owned, pid := t.process.owned, t.conn.pid
	return m.lendTo(f, store, ref, func(h uintptr, access uint32) (uintptr, error) { return owned.LendHandle(h, access, pid) },
		t.process.exited)
}

// freeWait bounds how long a fill waits for the sections lent before it to
// be freed: an engine that keeps a mapping (CPU tensors in place) holds its
// section past its load.
const freeWait = 10 * time.Second

// lendTo fills f's section from ref once the sections lent before it are
// freed, so one component's section is held at a time (KIT-MB4), and lends
// it and a lease pipe end through dup into the engine whose exit closes
// exited.
func (m *ModelMemory) lendTo(f VideoModelFile, store storage.Store, ref storage.Ref,
	dup func(h uintptr, access uint32) (uintptr, error), exited <-chan struct{}) (string, error) {
	m.waitFreed(freeWait)
	var release func()
	if m.Hold != nil {
		r, err := m.Hold(f.Size)
		if err != nil {
			return "", err
		}
		release = r
	}
	m.mu.Lock()
	m.live++
	m.mu.Unlock()
	freed := func() {
		if release != nil {
			release()
		}
		m.mu.Lock()
		m.live--
		if m.live == 0 && m.idle != nil {
			close(m.idle)
			m.idle = nil
		}
		m.mu.Unlock()
	}
	s, err := modelfiles.FillRef(context.Background(), m.Records, store, ref, freed)
	if err != nil {
		return "", err
	}
	// The owner's use ends here; the section lives while the engine's loan
	// does.
	defer s.Close()
	loan, err := s.LendTo(dup)
	if err != nil {
		return "", err
	}
	lease, err := modelbytes.NewLease()
	if err != nil {
		loan.End()
		return "", err
	}
	end, err := dup(lease.Writer().Fd(), pipeWriteAccess)
	lease.Lent()
	if err != nil {
		loan.End()
		return "", err
	}
	loan.EndWhen(lease.Done())
	if exited != nil {
		loan.EndWhen(exited)
	}
	u, err := loan.URI(end)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// waitFreed waits until no section this ModelMemory lent is held, or for
// within.
func (m *ModelMemory) waitFreed(within time.Duration) {
	m.mu.Lock()
	if m.live == 0 {
		m.mu.Unlock()
		return
	}
	if m.idle == nil {
		m.idle = make(chan struct{})
	}
	idle := m.idle
	m.mu.Unlock()
	select {
	case <-idle:
	case <-time.After(within):
		log.Printf("pytorch: a model section lent earlier is still held after %v; the next fill goes ahead", within)
	}
}

// archiveFailure names why an archive fill failed without the folder the
// error may carry.
func archiveFailure(err error) string {
	switch {
	case errors.Is(err, modelfiles.ErrDigest):
		return "its bytes differ from the pinned SHA-256; the archive sets them aside"
	case errors.Is(err, modelfiles.ErrSource):
		return "its bytes could not be read"
	case errors.Is(err, modelfiles.ErrRequested):
		return "the archive no longer holds it"
	}
	return "the section could not be made"
}
