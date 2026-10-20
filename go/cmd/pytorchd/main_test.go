package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func noEnv(string) string { return "" }

// PYTORCH-T2, T9, T11: models install refuses found files that are not the
// pinned set and records nothing; serve, engine check and models refuse a
// set, an option or a source that does not fit before anything starts.
func TestTorchCommandsRefuseBeforeAnythingStarts(t *testing.T) {
	from, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(from, "diffusion_pytorch_model_i2v.safetensors"), []byte("not the weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	state := t.TempDir()
	var out, errs bytes.Buffer
	status := run([]string{"models", "install", "--from", from, "--home", t.TempDir(), "--state", state}, &out, &errs, noEnv)
	if status != 1 || !strings.Contains(errs.String(), "not the pinned") {
		t.Fatalf("status %d, stderr %q", status, errs.String())
	}
	if entries, err := os.ReadDir(state); err != nil || len(entries) != 0 {
		t.Fatalf("a refused set left state %v: %v", entries, err)
	}
	errs.Reset()
	if status := run([]string{"models", "check", "--state", state}, &out, &errs, noEnv); status != 1 || !strings.Contains(errs.String(), "pytorchd models install") {
		t.Fatalf("check without a record: status %d, stderr %q", status, errs.String())
	}
	for _, args := range [][]string{
		{"serve", "--endpoint", "pytorch-test", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--video-model", "fastwan2.2-ti2v-5b", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--video-model", "flux2-klein-4b", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--image-model", "hummingbird-xt-i2v", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--video-model", "hummingbird-xt-i2v", "--video-model-option", "decode-dtype=f16", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--image-model", "flux2-klein-4b", "--video-model-option", "decode-dtype=bf16", "--state", state},
		{"models", "install", "--set", "fastwan2.2-ti2v-5b", "--from", from, "--state", state},
		{"models", "check", "--from", from, "--state", state},
		{"engine", "install", "--state", state},
		{"engine", "check", "--python", "python.exe", "--state", state},
		{"engine", "check", "--set", "flux2-klein-4b", "--python", from + `\python.exe`, "--source", from, "--state", state},
		{"engine", "check", "--set", "unreviewed", "--python", from + `\python.exe`, "--state", state},
	} {
		errs.Reset()
		if status := run(args, &out, &errs, noEnv); status != 2 {
			t.Fatalf("%q: status %d, stderr %q", args, status, errs.String())
		}
	}
}

// The help names every command and the engine's installed containment.
func TestHelpNamesTheCommands(t *testing.T) {
	var out, errs bytes.Buffer
	if status := run(nil, &out, &errs, noEnv); status != 0 {
		t.Fatalf("status %d", status)
	}
	for _, want := range []string{"pytorchd serve", "pytorchd engine check", "pytorchd models install", "pytorchd gpu list", "Windows AppContainer"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("help lacks %q", want)
		}
	}
}
