package torch

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-storage/go/layout"
)

func TestDefaultComponentRootSharesModelHostEngineDir(t *testing.T) {
	cache := accountDataFixture(t)
	pytorch, err := DefaultComponentRoot()
	if err != nil {
		t.Fatal(err)
	}
	// This is the native default entry point used by ModelHost startup and
	// engine installation. PyTorch keeps its unversioned installed spelling.
	modelhost, err := layout.DefaultEngineDir("llama.cpp", "fixture-release")
	if err != nil {
		t.Fatal(err)
	}
	if pytorch != filepath.Join(cache, "OpenAbstractions", "engines", "pytorch") || filepath.Base(modelhost) != "llama.cpp-fixture-release" {
		t.Fatalf("installed paths changed: pytorch=%q modelhost=%q", pytorch, modelhost)
	}
	for _, path := range []string{pytorch, modelhost} {
		if err := os.MkdirAll(path, 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{pytorch, modelhost} {
		if same, err := identity.SameFile(filepath.Dir(path), layout.EngineCacheRoot(cache)); err != nil || !same {
			t.Fatalf("different existing engines root: %q %v %v", path, same, err)
		}
	}
}

func TestDefaultComponentRootPreservesNativeError(t *testing.T) {
	unsetAccountData(t)
	_, nativeErr := layout.AccountData()
	got, err := DefaultComponentRoot()
	if nativeErr == nil || err == nil || got != "" || err.Error() != "no local application data folder; pass --root: "+nativeErr.Error() {
		t.Fatalf("native error changed: root=%q error=%v native=%v", got, err, nativeErr)
	}
}

// accountDataFixture points layout.AccountData at a fresh folder and returns
// it: %LOCALAPPDATA% on Windows, the data root under $HOME on macOS and
// $XDG_DATA_HOME on Linux.
func accountDataFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	switch runtime.GOOS {
	case "windows":
		t.Setenv("LOCALAPPDATA", dir)
		return dir
	case "darwin":
		t.Setenv("HOME", dir)
		return filepath.Join(dir, "Library", "Application Support", "openabstractions")
	default:
		t.Setenv("HOME", dir)
		t.Setenv("XDG_DATA_HOME", filepath.Join(dir, "data"))
		return filepath.Join(dir, "data", "openabstractions")
	}
}

// unsetAccountData leaves layout.AccountData no base to resolve.
func unsetAccountData(t *testing.T) {
	t.Helper()
	t.Setenv("LOCALAPPDATA", "")
	t.Setenv("HOME", "")
	t.Setenv("XDG_DATA_HOME", "")
}
