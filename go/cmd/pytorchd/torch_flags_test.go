package main

import (
	"testing"

	"github.com/openabstractions/abstraction-provider-pytorch/go/torch"
)

// models install names the `openabstractions models host` flags that
// declare the set it recorded: an image set by --image-model, a video set by
// --video-kind torch --video-model (live walkthrough 2026-10-02, D13, moved
// from modelhostd with the kind).
func TestInstallNamesTheSetsDeclarationFlags(t *testing.T) {
	for _, c := range []struct {
		set  torch.TorchModelSet
		want string
	}{
		{torch.Flux2Klein4BTorch, "--image-model flux2-klein-4b"},
		{torch.HummingbirdXTI2VTorch, "--video-kind torch --video-model hummingbird-xt-i2v"},
	} {
		if got := modelsHostFlags(c.set); got != c.want {
			t.Fatalf("%s: %q, want %q", c.set.Name, got, c.want)
		}
	}
}
