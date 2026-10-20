package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"github.com/openabstractions/abstraction-facade/go-core/bootstrap"
	"github.com/openabstractions/abstraction-identity/fileid"
	"github.com/openabstractions/abstraction-identity/listen"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func noEnv(string) string { return "" }

// PYTORCH-T2, T9, T11: models install refuses found files that are not the
// pinned set and records nothing; serve, engine check and models refuse a
// set, an option or a source that does not fit before anything starts.
func TestTorchCommandsRefuseBeforeAnythingStarts(t *testing.T) {
	source, err := fileid.Resolve(context.Background(), fileid.Self, t.TempDir(), fileid.Options{})
	if err != nil {
		t.Fatal(err)
	}
	from := source.Path()
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
		{"serve", "--endpoint", "pytorch-test", "--image-model", "flux2-klein-4b", "--image-model", "flux2-klein-4b", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--image-model", "flux2-klein-4b", "--image-model", "z-image-turbo", "--image-model", "flux2-klein-4b-compact", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--video-model", "hummingbird-xt-i2v", "--image-model", "flux2-klein-4b", "--image-model", "z-image-turbo", "--state", state},
		{"serve", "--endpoint", "pytorch-test", "--image-model", "flux2-klein-4b", "--image-model", "hummingbird-xt-i2v", "--state", state},
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

// Naming is exercised through a real owned socket/pipe; no provider engine runs.
func TestLeafEndpointNamingAndReceiving(t *testing.T) {
	parent, err := os.MkdirTemp("", "st5-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(parent); err != nil {
			t.Error(err)
		}
	})
	if runtime.GOOS != "windows" {
		t.Setenv("TMPDIR", parent)
	}
	t.Setenv("XDG_RUNTIME_DIR", "")
	name := fmt.Sprintf("st5-leaf-%d", os.Getpid())
	if address, err := serveEndpoint(name); address != "" || !errors.Is(err, bootstrap.ErrNoTrustedInstallation) {
		t.Fatalf("default name escaped test refusal: %q %v", address, err)
	}
	address := filepath.Join(parent, name+".sock")
	if runtime.GOOS == "windows" {
		address = `\\.\pipe\oa-test-` + name
	}

	if actual, err := serveEndpoint(address); err != nil || actual != address {
		t.Fatalf("explicit address changed: %q %v", actual, err)
	}
	for _, explicit := range []string{`/owned/path with spaces/name.sock`, `C:\owned\path with spaces\name.sock`, `\\.\pipe\owned.Name_test`} {
		if actual, err := serveEndpoint(explicit); err != nil || actual != explicit {
			t.Fatalf("explicit %q changed to %q: %v", explicit, actual, err)
		}
	}
	listener, err := listen.Listen(address)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stopListener := context.AfterFunc(ctx, func() { listener.Close() })
	defer stopListener()
	answered := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			answered <- err
			return
		}
		defer conn.Close()
		stop := context.AfterFunc(ctx, func() { conn.Close() })
		defer stop()
		var request [1]byte
		if _, err = io.ReadFull(conn, request[:]); err == nil {
			if request[0] != 'q' {
				err = fmt.Errorf("unexpected request %q", request[0])
			} else {
				_, err = conn.Write([]byte{'r'})
			}
		}
		answered <- err
	}()
	conn, err := listen.Dial(address)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	stopConn := context.AfterFunc(ctx, func() { conn.Close() })
	defer stopConn()
	if _, err = conn.Write([]byte{'q'}); err != nil {
		t.Fatal(err)
	}
	var reply [1]byte
	if _, err = io.ReadFull(conn, reply[:]); err != nil || reply[0] != 'r' {
		t.Fatalf("reply %q: %v", reply, err)
	}
	if err := <-answered; err != nil {
		t.Fatal(err)
	}
}
func TestLeafEndpointRefusesBeforeProviderWork(t *testing.T) {
	for _, name := range []string{"model.v1", "model_v1", "Model", "model name"} {
		var output, diagnostics bytes.Buffer
		code := run([]string{"serve", "--endpoint", name}, &output, &diagnostics, func(string) string { return "" })
		if code != 2 || !strings.Contains(diagnostics.String(), "[a-z0-9-]+") || !strings.Contains(diagnostics.String(), "explicit absolute endpoint path") || output.Len() != 0 {
			t.Fatalf("%q: code %d, output %q, diagnostics %q", name, code, output.String(), diagnostics.String())
		}
	}
	if runtime.GOOS != "windows" {
		// A short name resolves in the runtime directory below; test binaries
		// need the opt-in to resolve installed endpoints at all.
		t.Setenv("OA_TEST_ALLOW_INSTALLED_RUNTIME", "1")
		dir := t.TempDir()
		if err := os.Chmod(dir, 0755); err != nil {
			t.Fatal(err)
		}
		t.Setenv("XDG_RUNTIME_DIR", dir)
		var output, diagnostics bytes.Buffer
		code := run([]string{"serve", "--endpoint", "leaf-v1"}, &output, &diagnostics, func(string) string { return "" })
		if code != 2 || !strings.Contains(diagnostics.String(), "private") || output.Len() != 0 {
			t.Fatalf("namespace: code %d, output %q, diagnostics %q", code, output.String(), diagnostics.String())
		}
	}
}
