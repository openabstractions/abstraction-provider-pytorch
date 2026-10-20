package torch

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
)

// engineRuns skips a test that needs the engine running: the provider
// launches it only inside its AppContainer, and the reviewed stack is pinned
// for Windows x64 only (README "Platforms", sandbox_other.go refuses every
// placement elsewhere).
func engineRuns(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "windows" {
		t.Skip("the PyTorch engine runs only on Windows, inside its AppContainer")
	}
}

// The helpers below are the model host's test helpers the PyTorch tests
// used there, kept beside the tests that moved here.

// gpuEvents is a stand-in platform: what a test sends is a record.
type gpuEvents struct{ events chan gpufault.Event }

func (s gpuEvents) Watch(ctx context.Context, deliver func(gpufault.Event)) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		case ev := <-s.events:
			deliver(ev)
		}
	}
}

// gpuRig is one guard over a stand-in platform, a switchable probe, a
// counted wake lease and the faults it closes.
type gpuRig struct {
	guard   *gpufault.Guard
	source  gpuEvents
	faults  chan gpufault.Fault
	mu      sync.Mutex
	probeOK bool
	woken   int
	slept   int
}

func newGPURig(t *testing.T) *gpuRig {
	t.Helper()
	r := &gpuRig{source: gpuEvents{events: make(chan gpufault.Event)}, faults: make(chan gpufault.Fault, 4), probeOK: true}
	guard, err := gpufault.New(gpufault.Options{
		Records: r.source, Settle: 100 * time.Millisecond, Window: 2 * time.Second,
		Probe: func(context.Context) error {
			r.mu.Lock()
			defer r.mu.Unlock()
			if !r.probeOK {
				return gpufault.ErrNoAdapter
			}
			return nil
		},
		Wake: func(string) (func(), error) {
			r.mu.Lock()
			r.woken++
			r.mu.Unlock()
			return func() { r.mu.Lock(); r.slept++; r.mu.Unlock() }, nil
		},
		OnFault: func(f gpufault.Fault) { r.faults <- f },
	})
	if err != nil {
		t.Fatal(err)
	}
	r.guard = guard
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go guard.Run(ctx)
	return r
}

func (r *gpuRig) setProbe(ok bool) { r.mu.Lock(); r.probeOK = ok; r.mu.Unlock() }

func (r *gpuRig) wake() (int, int) { r.mu.Lock(); defer r.mu.Unlock(); return r.woken, r.slept }

// nextGPUFault is the next closed fault, or a failed test after 15 s.
func nextGPUFault(t *testing.T, faults <-chan gpufault.Fault) gpufault.Fault {
	t.Helper()
	select {
	case f := <-faults:
		return f
	case <-time.After(15 * time.Second):
		t.Fatal("no fault was recorded")
		return gpufault.Fault{}
	}
}

func waitUntil(t *testing.T, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatal("condition not reached")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// openACL stands in for the read-only folder ACL, which tests do not set on
// their temporary folders.
type openACL struct{}

func (openACL) Restrict([]string) error { return nil }

func (openACL) Check(string, *enginefiles.Handle) error { return nil }

// fakeLeases is the lease service as this provider sees it: it grants, it
// records renewals and releases, and it can put one yield request to the
// holder the way the book does.
type fakeLeases struct {
	mu            sync.Mutex
	outcome       res.AcquireOutcome
	err           error
	acquires      int
	renews        int
	released      []string
	answered      []string
	events        []string
	live          map[string]bool
	seq           int
	bindings      map[string]int
	bindOutcome   res.ProcessBindOutcome
	beforeRelease func()
	// resources is the resource of every Acquire, in order.
	resources []string
	// capacity, when not zero, is the resource's size: an Acquire beyond
	// what the live leases leave is insufficient. amounts and grants are
	// each live lease's amount and every granted amount in order.
	capacity int64
	amounts  map[string]int64
	grants   []int64

	pending chan res.YieldRequest
}

func newFakeLeases() *fakeLeases {
	return &fakeLeases{outcome: res.AcquireOutcomeAcquired, bindOutcome: res.ProcessBindOutcomeBound,
		live: map[string]bool{}, bindings: map[string]int{}, pending: make(chan res.YieldRequest, 4)}
}

func (f *fakeLeases) BindProcess(lease string, pid int) (res.ProcessBindResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bindings[lease] = pid
	return res.ProcessBindResult{Outcome: f.bindOutcome}, nil
}

func (f *fakeLeases) Acquire(resource string, amount int64, wait time.Duration) (res.AcquireResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return res.AcquireResult{}, f.err
	}
	f.acquires++
	f.resources = append(f.resources, resource)
	if f.outcome != res.AcquireOutcomeAcquired {
		return res.AcquireResult{Outcome: f.outcome, Asked: []res.YieldRecord{{Holder: "host:lmstudio", Answer: "refused: no rule"}}}, nil
	}
	if f.capacity > 0 {
		held := int64(0)
		for lease := range f.live {
			held += f.amounts[lease]
		}
		if held+amount > f.capacity {
			return res.AcquireResult{Outcome: res.AcquireOutcomeInsufficient, Asked: []res.YieldRecord{}}, nil
		}
	}
	f.seq++
	id := "lease-" + strconv.Itoa(f.seq)
	f.live[id] = true
	if f.amounts == nil {
		f.amounts = map[string]int64{}
	}
	f.amounts[id] = amount
	f.grants = append(f.grants, amount)
	return res.AcquireResult{Outcome: res.AcquireOutcomeAcquired,
		Lease: &res.Lease{ID: id, Resource: resource, Amount: amount}, Asked: []res.YieldRecord{}}, nil
}

func (f *fakeLeases) Renew(lease string) (res.LeaseChange, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.renews++
	return res.LeaseChange{Applied: f.live[lease]}, nil
}

func (f *fakeLeases) Release(lease string) (res.LeaseChange, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.beforeRelease != nil {
		f.beforeRelease()
	}
	f.released = append(f.released, lease)
	f.events = append(f.events, "release:"+lease)
	applied := f.live[lease]
	delete(f.live, lease)
	return res.LeaseChange{Applied: applied}, nil
}

func (f *fakeLeases) Observe(ctx context.Context, cursor string, wait time.Duration) (res.YieldPage, error) {
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case request := <-f.pending:
		return res.YieldPage{Requests: []res.YieldRequest{request}, Next: request.ID}, nil
	case <-timer.C:
		return res.YieldPage{Requests: []res.YieldRequest{}, Next: cursor}, nil
	case <-ctx.Done():
		return res.YieldPage{Requests: []res.YieldRequest{}, Next: cursor}, ctx.Err()
	}
}

func (f *fakeLeases) Answer(ctx context.Context, request string, answer res.YieldAnswer, reason string) (res.AnswerResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.answered = append(f.answered, request+"="+answer.String())
	f.events = append(f.events, "answer:"+request)
	return res.AnswerResult{Recorded: true}, nil
}

func (f *fakeLeases) counts() (acquires int, released []string, answered []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.acquires, append([]string(nil), f.released...), append([]string(nil), f.answered...)
}

var testCaller = inference.Subject{Account: "S-1-5-21-test", Program: `C:\OA\openabstractions.exe`}

// awaitVideo observes op until its status is one of want.
func awaitVideo(t *testing.T, h *Host, op string, want ...wire.VideoState) wire.VideoObservation {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for {
		observed := h.VideoServer().Observe(context.Background(), testCaller, op)
		if slices.Contains(want, observed.Status) {
			return observed
		}
		if time.Now().After(deadline) {
			t.Fatalf("operation %s is %s, waiting for %v", op, observed.Status, want)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func startVideo(t *testing.T, h *Host, request wire.VideoRequest) string {
	t.Helper()
	admission := h.VideoServer().Start(context.Background(), testCaller, request)
	if admission.Outcome != wire.StartOutcomeAccepted || admission.Operation == "" {
		t.Fatalf("admission %+v", admission)
	}
	return admission.Operation
}

func readFile(dir, name string) string {
	raw, _ := os.ReadFile(filepath.Join(dir, name))
	return string(raw)
}

// awaitUnloaded waits for the video engine to stop and returns whether it
// did within 30 seconds.
func awaitUnloaded(h *Host) bool { return awaitUnloadedWithin(h, 30*time.Second) }

// awaitUnloadedWithin waits up to within for the video engine to stop.
func awaitUnloadedWithin(h *Host, within time.Duration) bool {
	deadline := time.Now().Add(within)
	for h.Loaded(torchTestModel) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	return !h.Loaded(torchTestModel)
}

// engineExits collects the exit records a host reports (KIT-P3).
type engineExits struct {
	mu  sync.Mutex
	got []EngineExit
}

func (e *engineExits) hook(x EngineExit) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.got = append(e.got, x)
}

// await is the first record, once one arrives.
func (e *engineExits) await(t *testing.T) EngineExit {
	t.Helper()
	waitUntil(t, func() bool {
		e.mu.Lock()
		defer e.mu.Unlock()
		return len(e.got) > 0
	})
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.got[0]
}
