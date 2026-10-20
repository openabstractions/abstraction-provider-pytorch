package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
	torch "github.com/openabstractions/abstraction-provider-pytorch/go/torch"
)

func TestSetupRemovePassesOnlyExpectedManagedRootIdentity(t *testing.T) {
	old := componentRemoval
	t.Cleanup(func() { componentRemoval = old })
	root := filepath.Join(t.TempDir(), "uncreated")
	calls := 0
	componentRemoval = func(c torch.Component, _ io.Writer) (*sandbox.Removal, error) {
		calls++
		if c.Root != root || c.ExpectedRootKey != "trusted-journal-key" {
			t.Fatal("removal identity changed", c)
		}
		return nil, nil
	}
	var out, failure bytes.Buffer
	if code := runSetup([]string{"remove", "--root", root, "--expected-root-key", "trusted-journal-key"}, &out, &failure); code != 0 || calls != 1 {
		t.Fatal("managed remove dispatch", code, failure.String())
	}
	for _, args := range [][]string{{"inputs", "--expected-root-key="}, {"inputs", "--expected-root-key", "key"}, {"install", "--root", root, "--expected-root-key="}} {
		if code := runSetup(args, &out, &failure); code != 2 || calls != 1 {
			t.Fatal("unexpected removal/input effect", code, failure.String())
		}
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatal("dispatch created component root", err)
	}
}
