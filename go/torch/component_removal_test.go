package torch

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
)

// Sandbox replies are stubs. These ordinary lifecycle fixtures never create
// a profile, apply a grant or launch an engine.
func TestManagedComponentRemovalInterruptionOwnership(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("component placement is Windows-only")
	}
	for _, kind := range []string{"before-install", "after-sandbox-remove", "profile-present", "profile-query-failed", "keyless", "replaced", "missing-root", "foreign-extra"} {
		t.Run(kind, func(t *testing.T) {
			c := Component{Root: filepath.Join(t.TempDir(), "generated")}
			if err := os.MkdirAll(c.EngineDir(), 0700); err != nil {
				t.Fatal(err)
			}
			payload := filepath.Join(c.EngineDir(), "fixture.txt")
			if err := os.WriteFile(payload, []byte("generated fixture"), 0600); err != nil {
				t.Fatal(err)
			}
			key, err := identity.FileKey(c.Root)
			if err != nil {
				t.Fatal(err)
			}
			c.ExpectedRootKey = key
			profile, err := c.Profile()
			if err != nil {
				t.Fatal(err)
			}
			if kind == "keyless" {
				c.ExpectedRootKey = ""
			}
			if kind == "missing-root" {
				if err := os.RemoveAll(c.Root); err != nil {
					t.Fatal(err)
				}
			}
			if kind == "replaced" {
				if err := os.Rename(c.Root, c.Root+"-original"); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(c.Root, 0700); err != nil {
					t.Fatal(err)
				}
			}
			personal := filepath.Join(c.Root, "personal.txt")
			if kind == "foreign-extra" {
				if err := os.WriteFile(personal, []byte("personal"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			events := []string{}
			removeSandbox := func(path string, keep bool) (*sandbox.Removal, error) {
				if path != c.Manifest() || keep {
					t.Fatal("changed sandbox authority")
				}
				events = append(events, "sandbox")
				return nil, sandbox.ErrNotInstalled
			}
			profiles := func(prefix string) ([]string, error) {
				if prefix != profile {
					t.Fatal("changed derived profile")
				}
				events = append(events, "profiles")
				if kind == "profile-present" {
					return []string{profile}, nil
				}
				if kind == "profile-query-failed" {
					return nil, errors.New("fixture inventory unavailable")
				}
				return nil, nil
			}
			removeEngine := func(path string) error {
				if path != c.EngineDir() {
					t.Fatal("removed foreign tree")
				}
				events = append(events, "engine")
				return removeTree(path)
			}
			if kind == "after-sandbox-remove" {
				// The sandbox owner completed its phase, then the generated tree
				// step failed. The next call has the same key and no manifest.
				firstSandbox := func(string, bool) (*sandbox.Removal, error) {
					events = append(events, "sandbox")
					return &sandbox.Removal{Profile: profile, ProfileDeleted: true}, nil
				}
				firstTree := func(string) error {
					events = append(events, "tree-interrupted")
					return errors.New("fixture removal interrupted")
				}
				if _, err := removeComponent(c, io.Discard, firstSandbox, profiles, firstTree); err == nil {
					t.Fatal("incomplete removal claimed success")
				}
				if current, err := identity.FileKey(c.Root); err != nil || current != key {
					t.Fatal("interruption changed root", err)
				}
				events = nil
			}
			_, err = removeComponent(c, io.Discard, removeSandbox, profiles, removeEngine)
			success := kind == "before-install" || kind == "after-sandbox-remove"
			if (err == nil) != success {
				t.Fatalf("%s: %v", kind, err)
			}
			if success {
				if !slices.Equal(events, []string{"sandbox", "profiles", "engine"}) {
					t.Fatal("wrong cleanup order", events)
				}
				if _, err := os.Stat(c.Root); !errors.Is(err, os.ErrNotExist) {
					t.Fatal("generated root retained", err)
				}
			} else if kind == "foreign-extra" {
				if data, err := os.ReadFile(personal); err != nil || string(data) != "personal" {
					t.Fatal("foreign root data changed", err)
				}
			} else if slices.Contains(events, "engine") {
				t.Fatal("unproven cleanup ran", events)
			}
		})
	}
}
