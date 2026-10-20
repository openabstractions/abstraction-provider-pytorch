package main

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/openabstractions/abstraction-storage/go/layout"
)

// ST-2 ruling 2a: the provider's default state folder lies under
// layout.AccountData (the data root on macOS and Linux).
func TestDefaultStatePathFollowsAccountData(t *testing.T) {
	home := t.TempDir()
	if runtime.GOOS == "windows" {
		t.Setenv("LOCALAPPDATA", home)
	} else {
		t.Setenv("HOME", home)
		t.Setenv("XDG_DATA_HOME", filepath.Join(home, "data"))
	}
	data, err := layout.AccountData()
	if err != nil {
		t.Fatal(err)
	}
	got, err := statePath("")
	if want := filepath.Join(data, "OpenAbstractions", "providers", "pytorch"); err != nil || got != want {
		t.Fatalf("default state %q %v, want %q", got, err, want)
	}
}
