package torch

import (
	"context"
	"strings"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
)

// RES-G5: standby that begins while the engine loads the set refuses the
// image before its first dispatch: no generate reaches the engine, the
// reply is gpu:standby at the bound, and the job ends its use. Once
// standby ends the next image is admitted.
func TestTorchStandbyDuringLoadSendsNoGenerate(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--image-mode=slow-load", "--load-seconds=2")
	r := newGPURig(t)
	leases := newFakeLeases()
	h := f.imageHost(t, leases, func(o *Options) { o.GPU = r.guard })
	old := gpuWaitBound
	gpuWaitBound = 300 * time.Millisecond
	t.Cleanup(func() { gpuWaitBound = old })
	go func() {
		for deadline := time.Now().Add(30 * time.Second); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
			if strings.Contains(f.log(), `"event": "load"`) {
				break
			}
		}
		r.source.events <- gpufault.Event{Kind: gpufault.StandbyEnter, Source: "Kernel-Power 506"}
	}()
	reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse"))
	if reply.outcome != wire.ReplyOutcomeUnavailable || reply.reason != gpufault.WordStandby {
		t.Fatalf("reply %+v; log %s", reply, f.log())
	}
	for _, rec := range f.records(t) {
		if rec["event"] == "generate" {
			t.Fatalf("a generate reached the engine in standby: %v", rec)
		}
	}
	// The loaded set keeps its one card lease for the next image; the
	// refused job asked for no second one.
	if acquires, _, _ := leases.counts(); acquires != 1 {
		t.Fatalf("acquires %d", acquires)
	}
	r.source.events <- gpufault.Event{Kind: gpufault.StandbyExit, Source: "Kernel-Power 507"}
	waitUntil(t, func() bool { return r.guard.AdmitLoad(context.Background()) == nil })
	if again, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); again.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("the image after standby: %+v; log %s", again, f.log())
	}
}
