package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity/go/privatedir"
	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"

	"github.com/openabstractions/abstraction-provider-pytorch/go/torch"
)

// stageFromJobStore registers a stand-in set whose files the job service
// staged under root\set: each file's first name lies in the runtime's job
// store under base, as LinkResult leaves it (JOB-A17). With extra, each file
// also gets a name under base\setup-state, outside every OA root.
func stageFromJobStore(t *testing.T, extra bool) (base, root, from, state string, getenv func(string) string, set torch.TorchModelSet) {
	t.Helper()
	saved := torch.TorchModelSets
	t.Cleanup(func() { torch.TorchModelSets = saved })
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	getenv = func(name string) string {
		if name == "LOCALAPPDATA" {
			return base
		}
		return ""
	}
	results := filepath.Join(modelplace.DefaultJobStore(getenv), "results")
	root = filepath.Join(base, "OA", "m")
	from = filepath.Join(root, "set")
	state = filepath.Join(base, "OA", "p", "pt")
	setup := filepath.Join(base, "setup-state")
	for _, dir := range []string{results, from, state, setup} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	set = torch.HummingbirdXTI2VTorch
	set.Name = "jobstore-link-test"
	set.Files = append([]torch.VideoModelFile(nil), set.Files...)
	for i, f := range set.Files {
		data := []byte("stand-in " + f.Name)
		result := filepath.Join(results, fmt.Sprintf("result-%d", i))
		if err := os.WriteFile(result, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Link(result, filepath.Join(from, f.Name)); err != nil {
			t.Skipf("this volume makes no hard links: %v", err)
		}
		if extra {
			if err := os.Link(result, filepath.Join(setup, fmt.Sprintf("queued-%d", i))); err != nil {
				t.Fatal(err)
			}
		}
		sum := sha256.Sum256(data)
		set.Files[i].Size, set.Files[i].SHA256, set.Files[i].GitBlobSHA1 = int64(len(data)), hex.EncodeToString(sum[:]), ""
	}
	torch.TorchModelSets = append(append([]torch.TorchModelSet(nil), saved...), set)
	return base, root, from, state, getenv, set
}

// The owner's R18 Setup stuck on "Preparing…": every staged file had a
// second name in the job store and Setup passed --job-store "". models
// install admits the runtime's job store as one of OA's own roots without
// the flag (modelplace.OwnedRoots, KIT-M3).
func TestModelsInstallAdmitsTheJobStoreWithoutTheFlag(t *testing.T) {
	_, root, from, state, getenv, set := stageFromJobStore(t, false)
	var out, errs bytes.Buffer
	status := run([]string{"models", "install", "--set", set.Name, "--from", from, "--models-root", root, "--state", state}, &out, &errs, getenv)
	if status != 0 {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out.String(), errs.String())
	}
	if err := privatedir.CheckModelFolder(from); err != nil {
		t.Fatalf("the staged set's folder is not a model folder: %v", err)
	}
}

// A name outside OA's own roots, such as a link under Setup's state, still
// refuses the folder unchanged (identity ID-D3).
func TestModelsInstallRefusesALinkOutsideOwnedRoots(t *testing.T) {
	_, root, from, state, getenv, set := stageFromJobStore(t, true)
	var out, errs bytes.Buffer
	status := run([]string{"models", "install", "--set", set.Name, "--from", from, "--models-root", root, "--state", state}, &out, &errs, getenv)
	if status != 1 || !strings.Contains(errs.String(), "not a model folder") {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out.String(), errs.String())
	}
	if privatedir.CheckModelFolder(from) == nil {
		t.Fatal("a folder with a name outside OA's roots became a model folder")
	}
}

// An empty --job-store once meant "no job store" without complaint; it is
// refused before anything is placed.
func TestModelsRefuseAnEmptyJobStore(t *testing.T) {
	_, root, from, state, getenv, set := stageFromJobStore(t, false)
	for _, verb := range []string{"install", "protect"} {
		var out, errs bytes.Buffer
		args := []string{"models", verb, "--set", set.Name, "--models-root", root, "--state", state, "--job-store", ""}
		if verb == "install" {
			args = append(args, "--from", from)
		}
		if status := run(args, &out, &errs, getenv); status != 2 || !strings.Contains(errs.String(), "job-store") {
			t.Fatalf("%s: status %d, stderr %q", verb, status, errs.String())
		}
	}
	if privatedir.CheckModelFolder(from) == nil {
		t.Fatal("a refused command protected the folder")
	}
}
