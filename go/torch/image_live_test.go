package torch

import (
	"bytes"
	"context"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// TestLiveTorchImageOnTheCard makes one FLUX.2 [klein] 4B image on the real
// GPU through the shipped engine: the component `pytorchd setup install`
// made from the installer's input bundle, the model files `pytorchd models
// install` placed, the engine's AppContainer and the set's job memory limit.
// It is the owner's GPU smoke test and runs only with OA_LIVE_PYTORCH_IMAGE=1
// under research/engines/pytorch-image-smoke.ps1, which tells the owner
// first, watches for a driver reset and stops the run at the first one
// (owner rule, 2026-10-01). Leases are this test's own: the card must be
// idle. OA_LIVE_PYTORCH_ROOT is the component root, OA_LIVE_PYTORCH_STATE
// the provider state the models were installed into and OA_LIVE_PYTORCH_OUT
// the PNG it writes. OA_LIVE_PYTORCH_SET names another image set, such as
// z-image-turbo.
func TestLiveTorchImageOnTheCard(t *testing.T) {
	if os.Getenv("OA_LIVE_PYTORCH_IMAGE") != "1" {
		t.Skip("a GPU run; set OA_LIVE_PYTORCH_IMAGE=1 through research/engines/pytorch-image-smoke.ps1")
	}
	engineRuns(t)
	root, state, out := os.Getenv("OA_LIVE_PYTORCH_ROOT"), os.Getenv("OA_LIVE_PYTORCH_STATE"), os.Getenv("OA_LIVE_PYTORCH_OUT")
	for _, path := range []string{root, state, out} {
		if !filepath.IsAbs(path) {
			t.Fatalf("OA_LIVE_PYTORCH_ROOT, _STATE and _OUT must be absolute paths, got %q", path)
		}
	}
	component := Component{Root: root}
	set := Flux2Klein4BTorch
	if name := os.Getenv("OA_LIVE_PYTORCH_SET"); name != "" {
		var ok bool
		if set, ok = TorchModelSetNamed(name); !ok || set.Profile != torchProfileImage {
			t.Fatalf("OA_LIVE_PYTORCH_SET %q is no image set", name)
		}
	}
	engine := &TorchEngine{Python: component.Python(), Set: set,
		Places: TorchModelPlaces(state, set), ModelRecord: TorchModelRecord(state, set),
		State: filepath.Join(state, "torch"), Ready: 10 * time.Minute, Log: filepath.Join(state, "logs", "torch.log"),
		MemoryLimit: set.Commit, Sandbox: component.Manifest()}
	h, err := New(Options{Leases: newFakeLeases(), Idle: time.Minute, Wait: time.Minute,
		Image: &ImageOptions{Model: set.Name, Torch: engine}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	go h.Run(ctx)
	image, route := h.ImageEngine()
	if image == nil || route == nil {
		t.Fatal("the host serves no image engine")
	}
	start := time.Now()
	request := wire.ImageRequest{Model: set.Name, Mode: wire.ImageModeGenerate, Prompt: "a lighthouse on a cliff at dusk, oil painting", Size: "auto", Count: 1}
	reply := image.Run(ctx, set.Name, request, nil, func(percent int64, phase string) {
		t.Logf("%6.1f s %3d%% %s", time.Since(start).Seconds(), percent, phase)
	})
	if reply.Outcome != wire.ReplyOutcomeCompleted || len(reply.Images) != 1 {
		t.Fatalf("reply %s %q after %.1f s; engine log %s", reply.Outcome, reply.Reason, time.Since(start).Seconds(), engine.Log)
	}
	picture, err := png.Decode(bytes.NewReader(reply.Images[0].Data))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, reply.Images[0].Data, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("first image %v in %.1f s (load included), written to %s", picture.Bounds(), time.Since(start).Seconds(), out)
}
