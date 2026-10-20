package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity/go/privatedir"

	"github.com/openabstractions/abstraction-provider-pytorch/go/torch"
)

// Delta review of the PyTorch engine kind (coordinator, 2026-10-02), moved
// with the kind: models install adopts the set folder it placed files in
// under the models root; a sibling folder under the root keeps its access
// (KIT-M3, identity ID-D3).
func TestModelsInstallAdoptsTheSetFolder(t *testing.T) {
	saved := torch.TorchModelSets
	t.Cleanup(func() { torch.TorchModelSets = saved })
	base := t.TempDir()
	root := filepath.Join(base, "OA-models")
	from := filepath.Join(root, "set")
	sibling := filepath.Join(root, "sibling")
	state := filepath.Join(base, "state")
	for _, dir := range []string{from, sibling, state} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(sibling, "kept.bin"), []byte("kept"), 0o600); err != nil {
		t.Fatal(err)
	}
	set := torch.HummingbirdXTI2VTorch
	set.Name = "adopt-test-torch"
	// The stand-in sizes and digests go on a copy: the pinned set's Files
	// slice is shared.
	set.Files = append([]torch.VideoModelFile(nil), set.Files...)
	for i, f := range set.Files {
		data := []byte("stand-in " + f.Name)
		if err := os.WriteFile(filepath.Join(from, f.Name), data, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		set.Files[i].Size, set.Files[i].SHA256, set.Files[i].GitBlobSHA1 = int64(len(data)), hex.EncodeToString(sum[:]), ""
	}
	torch.TorchModelSets = append(append([]torch.TorchModelSet(nil), saved...), set)
	var out, errs bytes.Buffer
	status := run([]string{"models", "install", "--set", set.Name, "--from", from, "--models-root", root,
		"--home", filepath.Join(base, "home"), "--state", state, "--job-store", filepath.Join(base, "jobs")}, &out, &errs, noEnv)
	if status != 0 {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out.String(), errs.String())
	}
	if !strings.Contains(out.String(), from+": OA's model folder, protected") ||
		!strings.Contains(out.String(), "declare it with openabstractions models host --video-kind torch --video-model adopt-test-torch\n") {
		t.Fatalf("stdout %q", out.String())
	}
	if err := privatedir.CheckModelFolder(from); err != nil {
		t.Fatalf("the installed set's folder is not a model folder: %v", err)
	}
	if privatedir.CheckModelFolder(sibling) == nil {
		t.Fatal("the sibling folder became a model folder")
	}
}
