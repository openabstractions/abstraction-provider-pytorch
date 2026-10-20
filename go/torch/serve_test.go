package torch

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// PYTORCH-T11: the served configuration gives the inference provider the
// image engine for exactly its router host, Admit lets the image model
// through without loading anything (the image set loads under the holder
// when it runs), and the published bound is 20 MiB for images alone.
func TestTheServedConfigurationCarriesTheImageEngine(t *testing.T) {
	state := t.TempDir()
	set := Flux2Klein4BTorch
	engine := &TorchEngine{Python: filepath.Join(state, "venv", "Scripts", "python.exe"), Set: set,
		Places: TorchModelPlaces(state, set), ModelRecord: TorchModelRecord(state, set), State: filepath.Join(state, "torch")}
	h, err := New(Options{Leases: newFakeLeases(), Idle: time.Hour, Image: &ImageOptions{Torch: engine}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	cfg := ProviderConfig(h, nil)
	images, route := h.ImageEngine()
	if cfg.ImageEngine == nil || cfg.ImageEngine != images || cfg.ImageEngineHost != route || route == nil {
		t.Fatalf("image engine %v host %v", cfg.ImageEngine, cfg.ImageEngineHost)
	}
	if !route.ServesModel("image", set.Name) || route.ServesModel("chat", set.Name) {
		t.Fatal("the image route serves the wrong profiles")
	}
	if word, ok := Admit(h)(context.Background(), route, set.Name); !ok || word != "" {
		t.Fatalf("admit %q %v", word, ok)
	}
	if h.Loaded(set.Name) {
		t.Fatal("Admit loaded the image set")
	}
	if word, ok := Admit(h)(context.Background(), route, "tiny"); ok || word != "model:undeclared" {
		t.Fatalf("admit of another model %q %v", word, ok)
	}
	if limit, ok := ImageOutputLimit("image", "image/*"); !ok || limit != 20<<20 {
		t.Fatalf("image limit %d %v", limit, ok)
	}
	if _, ok := ImageOutputLimit("speech", "audio/wav"); ok {
		t.Fatal("a speech output would be published")
	}
}

// PYTORCH-T11: the served provider routes an image request for its set to
// its own image engine host. A provider whose router holds no host answers
// every image request no_host "router:not-here", which is what the
// installed pytorchd answered the runtime on 2026-11-09. The request asks
// two images, which the engine refuses after routing and before any lease
// or engine start.
func TestTheServedProviderRoutesItsImageSet(t *testing.T) {
	state := t.TempDir()
	set := Flux2Klein4BTorch
	engine := &TorchEngine{Python: filepath.Join(state, "venv", "Scripts", "python.exe"), Set: set,
		Places: TorchModelPlaces(state, set), ModelRecord: TorchModelRecord(state, set), State: filepath.Join(state, "torch")}
	leases := newFakeLeases()
	h, err := New(Options{Leases: leases, Idle: time.Hour, Image: &ImageOptions{Torch: engine}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	expectRoutedImage(t, h, leases, set.Name)
}

// PYTORCH-T10, PYTORCH-T11: pytorchd with a video set serves the image set
// beside it in the video engine's process (pytorchd's Beside). The image
// request routes to the image set's engine host there too.
func TestTheServedProviderRoutesTheImageSetBesideTheVideoSet(t *testing.T) {
	state := t.TempDir()
	image := Flux2Klein4BTorch
	video := HummingbirdXTI2VTorch
	store, err := storage.NewContentStore("pytorch-video", filepath.Join(state, "content"))
	if err != nil {
		t.Fatal(err)
	}
	engine := &TorchEngine{Python: filepath.Join(state, "venv", "Scripts", "python.exe"), Set: video,
		Places: TorchModelPlaces(state, video), ModelRecord: TorchModelRecord(state, video), State: filepath.Join(state, "torch"),
		Beside: &TorchSetPlace{Set: image, Places: TorchModelPlaces(state, image), ModelRecord: TorchModelRecord(state, image)}}
	leases := newFakeLeases()
	h, err := New(Options{Leases: leases, Idle: time.Hour, Video: &VideoOptions{Torch: engine, Store: store}, Image: &ImageOptions{}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	expectRoutedImage(t, h, leases, image.Name)
}

// expectRoutedImage composes pytorchd's inference provider for h and asks
// it for two images of model. A provider whose router holds no host answers
// no_host "router:not-here", which is what the installed pytorchd answered
// the runtime on 2026-11-09. The image engine refuses the count after
// routing, before any lease or engine start.
func expectRoutedImage(t *testing.T, h *Host, leases *fakeLeases, model string) {
	t.Helper()
	provider, err := inference.New(ProviderConfig(h, nil))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = provider.Close() })
	caller := inference.Subject{Account: "S-1-5-21-fixture", Program: `C:\apps\opencode.exe`}
	a := provider.StartImage(context.Background(), caller, wire.ImageRequest{Model: model, Mode: wire.ImageModeGenerate,
		Prompt: "a lighthouse", Size: "auto", Count: 2, Guarantees: []wire.RequestGuarantee{wire.RequestGuaranteeLocalOnly}})
	if a.Outcome == wire.StartOutcomeNoHost || a.Outcome != wire.StartOutcomeUnsupportedFeature {
		t.Fatalf("the served provider did not route its image set to its engine: %+v", a)
	}
	if h.Loaded(model) || len(leases.grants) != 0 {
		t.Fatal("a refused image loaded the set")
	}
}
