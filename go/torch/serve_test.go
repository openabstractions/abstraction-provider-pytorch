package torch

import (
	"context"
	"path/filepath"
	"testing"
	"time"
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
