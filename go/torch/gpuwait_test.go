package torch

import (
	"context"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-resource/go/gpufault"
)

func standbyRig(t *testing.T) *gpuRig {
	t.Helper()
	r := newGPURig(t)
	r.source.events <- gpufault.Event{Kind: gpufault.StandbyEnter, Source: "Kernel-Power 506"}
	waitUntil(t, func() bool { return r.guard.AdmitLoad(context.Background()) != nil })
	return r
}

// An image held by standby is refused gpu:standby at the wait bound.
func TestImageGPUWaitEndsAtBound(t *testing.T) {
	r := standbyRig(t)
	old := gpuWaitBound
	gpuWaitBound = 300 * time.Millisecond
	t.Cleanup(func() { gpuWaitBound = old })
	start := time.Now()
	word := admitImageGPU(context.Background(), r.guard, gpufault.Fingerprint{}, func(int64, string) {})
	if word != gpufault.WordStandby {
		t.Fatalf("word %q", word)
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Fatalf("waited %s", d)
	}
}

// An image whose own deadline passes in standby is refused gpu:standby.
func TestImageGPUWaitEndsAtRequestDeadline(t *testing.T) {
	r := standbyRig(t)
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if word := admitImageGPU(ctx, r.guard, gpufault.Fingerprint{}, func(int64, string) {}); word != gpufault.WordStandby {
		t.Fatalf("word %q", word)
	}
}

// An image whose standby clears before the bound proceeds.
func TestImageGPUWaitProceedsAfterWake(t *testing.T) {
	r := standbyRig(t)
	go func() {
		time.Sleep(300 * time.Millisecond)
		r.source.events <- gpufault.Event{Kind: gpufault.StandbyExit, Source: "Kernel-Power 507"}
	}()
	if word := admitImageGPU(context.Background(), r.guard, gpufault.Fingerprint{}, func(int64, string) {}); word != "" {
		t.Fatalf("word %q", word)
	}
}
