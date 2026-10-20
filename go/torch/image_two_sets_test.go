package torch

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// One image-only engine whose Beside is a second image set serves both as
// image@1 on one router host: Setup registers FLUX.2 [klein] and
// Z-Image-Turbo this way (owner report R21, 2026-11-10).
func TestOneEngineServesTwoImageSets(t *testing.T) {
	if err := ZImageTurboTorch.valid(); err != nil {
		t.Fatal(err)
	}
	state := t.TempDir()
	klein, zimage := Flux2Klein4BCompactTorch, ZImageTurboTorch
	engine := &TorchEngine{Python: filepath.Join(state, "venv", "Scripts", "python.exe"), Set: klein,
		Places: TorchModelPlaces(state, klein), ModelRecord: TorchModelRecord(state, klein), State: filepath.Join(state, "torch"),
		Beside: &TorchSetPlace{Set: zimage, Places: TorchModelPlaces(state, zimage), ModelRecord: TorchModelRecord(state, zimage)}}
	h, err := New(Options{Leases: newFakeLeases(), Idle: time.Hour, Image: &ImageOptions{Torch: engine}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	images, route := h.ImageEngine()
	if images == nil || route == nil {
		t.Fatal("no image engine")
	}
	for _, set := range []TorchModelSet{klein, zimage} {
		if !route.ServesModel("image", set.Name) || !h.ServesImage(set.Name) {
			t.Fatalf("the image route does not serve %s", set.Name)
		}
		if word, ok := Admit(h)(context.Background(), route, set.Name); !ok || word != "" {
			t.Fatalf("admit %s: %q %v", set.Name, word, ok)
		}
		request := wire.ImageRequest{Model: set.Name, Mode: wire.ImageModeGenerate, Prompt: "a lighthouse", Size: "auto", Count: 1}
		if got := images.Unsupported(request); got != "" {
			t.Fatalf("%s refused: %q", set.Name, got)
		}
	}
	// Each set keeps its own sizes: 1024x1024 is Z-Image-Turbo's preset and
	// no size of FLUX.2 [klein].
	big := wire.ImageRequest{Mode: wire.ImageModeGenerate, Prompt: "a lighthouse", Size: "1024x1024", Count: 1}
	big.Model = zimage.Name
	big.Prompt = "a lighthouse\non the coast"
	if got := images.Unsupported(big); got != "" {
		t.Fatalf("z-image 1024x1024 refused: %q", got)
	}
	big.Model = klein.Name
	if got := images.Unsupported(big); got != "size:512x512" {
		t.Fatalf("klein 1024x1024: %q", got)
	}
	other := wire.ImageRequest{Model: "tiny", Mode: wire.ImageModeGenerate, Prompt: "x", Size: "auto", Count: 1}
	if got := images.Unsupported(other); got != "model:tiny" {
		t.Fatalf("another model: %q", got)
	}
	if reply := images.Run(context.Background(), "tiny", other, nil, nil); reply.Outcome != wire.ReplyOutcomeUnsupportedFeature {
		t.Fatalf("another model ran: %+v", reply)
	}
	// Both sets measured in one process: the second set's lease is what
	// both hold less the first one's, and the job limit is both commits.
	if lease := h.torch.leaseFor(TorchSetPlace{Set: zimage}, klein.Name); lease != gibTenths(310+185)-klein.Lease {
		t.Fatalf("joined lease %d", lease)
	}
	if limit := TorchCommitTogether(klein, nil, zimage); limit != (40+25)<<30 {
		t.Fatalf("commit together %d", limit)
	}
}
