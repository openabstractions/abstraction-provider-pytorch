package torch

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const componentSecurityProbeOptIn = "OA_PYTORCH_SANDBOX_SECURITY_TESTS"

func requireComponentSecurityProbeOptIn(t *testing.T) {
	t.Helper()
	if os.Getenv(componentSecurityProbeOptIn) != "1" {
		t.Skipf("sandbox security probes are paused until the owner confirms Daybreak access is clear; set %s=1 only after that confirmation", componentSecurityProbeOptIn)
	}
}

// PYTORCH-T1: the network-capable packages diffusers and transformers import
// are stubs the engine's deny finder answers. In an interpreter with none of
// them installed, each imports; the version, constants and decorators an
// importer reads at import time are given; every call raises OANetworkDenied
// naming it; importlib.metadata reports the pinned versions a dependency
// check reads; a version-only distribution reports its version and has no
// module; and every name read is recorded.
func TestEngineAnswersNetworkPackagesWithStubs(t *testing.T) {
	requireComponentSecurityProbeOptIn(t)
	python, err := basePython()
	if err != nil {
		t.Skipf("no Python for the engine script: %v", err)
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "oa_torch_engine.py")
	if err := os.WriteFile(script, torchEngineScript, 0o600); err != nil {
		t.Fatal(err)
	}
	driver, err := filepath.Abs(filepath.Join("testdata", "faketorch", "network_stubs.py"))
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(python, "-I", "-S", "-B", driver, script)
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			t.Fatalf("driver: %v\n%s", err, exit.Stderr)
		}
		t.Fatal(err)
	}
	var got struct {
		Imports        map[string]string
		Calls          map[string]string
		Versions       map[string]*string
		Values         map[string]any
		FilelockImport string `json:"filelock_import"`
		Stdlib         map[string]string
		Uses           []string
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("driver output %q: %v", out, err)
	}
	for name, file := range got.Imports {
		if !strings.HasPrefix(file, "<oa network stub ") {
			t.Fatalf("%s imported from %q, not the stub", name, file)
		}
	}
	for call, said := range got.Calls {
		if !strings.Contains(said, "is a network call this engine refuses") {
			t.Fatalf("%s: %q, want OANetworkDenied", call, said)
		}
	}
	if got.Values["version"] != "0.36.2" || got.Values["offline"] != true || got.Values["decorator_passes_through"] != true ||
		got.Values["subclass_is_exception"] != true || got.Values["progress_bars_disabled"] != true ||
		strings.Contains(got.Values["endpoint"].(string), "huggingface") {
		t.Fatalf("values %v", got.Values)
	}
	for dist, want := range map[string]string{"huggingface-hub": "0.36.2", "huggingface_hub": "0.36.2", "requests": "2.34.2",
		"filelock": "4.0.8", "fsspec": "2026.9.0"} {
		if v := got.Versions[dist]; v == nil || *v != want {
			t.Fatalf("version of %s: %v, want %s", dist, v, want)
		}
	}
	// A system proxy on 127.0.0.1 would carry a loopback request out: the
	// standard library's HTTP clients are refused before they connect.
	if len(got.Stdlib) != 3 {
		t.Fatalf("standard-library checks %v", got.Stdlib)
	}
	for call, said := range got.Stdlib {
		if !strings.Contains(said, "oa-torch-engine: urllib.Request refused") && !strings.Contains(said, "oa-torch-engine: http.client.connect refused") {
			t.Fatalf("%s: %q, want the audit hook's refusal", call, said)
		}
	}
	if got.FilelockImport != "missing" {
		t.Fatalf("filelock, a version-only distribution, %s", got.FilelockImport)
	}
	for _, name := range []string{"huggingface_hub.hf_hub_download", "huggingface_hub.utils.validate_hf_hub_args", "requests.get"} {
		found := false
		for _, u := range got.Uses {
			found = found || u == name
		}
		if !found {
			t.Fatalf("%s was read and not recorded: %v", name, got.Uses)
		}
	}
}
