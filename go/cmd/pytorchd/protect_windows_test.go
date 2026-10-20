package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity/go/privatedir"
	"github.com/openabstractions/abstraction-identity/go/trustedpath"

	"github.com/openabstractions/abstraction-provider-pytorch/go/torch"
)

// Coordinator decision (b), 2026-10-02: a PyTorch set placed under the
// models root is OA's set folder like an sd-server set's. Its places record
// makes the folder a recorded set, models protect makes it a model folder,
// and its files then pass the trusted-in-place check for the owners the
// record keeps.
func TestTorchSetFolderIsProtectedAndTrusted(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "OA-models")
	state := filepath.Join(base, "state")
	folder := filepath.Join(root, "wan-diffusers")
	for _, dir := range []string{folder, state} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	set := torch.HummingbirdXTI2VTorch
	owners := torch.FolderSearch("").Owners()
	var lines strings.Builder
	for _, f := range set.Files {
		p := filepath.Join(folder, f.Name)
		if err := os.WriteFile(p, []byte("stand-in"), 0o600); err != nil {
			t.Fatal(err)
		}
		account := ""
		if owners.Account {
			account = ";account"
		}
		lines.WriteString(f.Name + "\t" + p + "\t" + strings.Join(owners.SIDs, ",") + account + "\n")
	}
	if err := os.WriteFile(torch.TorchModelPlaces(state, set), []byte(lines.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	recorded, err := torch.RecordedSetFolders(root, state)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, dir := range recorded {
		found = found || strings.EqualFold(dir, folder)
	}
	if !found {
		t.Fatalf("the torch set's folder is not recorded: %v", recorded)
	}
	var out, errs bytes.Buffer
	if status := run([]string{"models", "protect", "--models-root", root, "--state", state, "--job-store", filepath.Join(base, "jobs")}, &out, &errs, noEnv); status != 0 {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out.String(), errs.String())
	}
	if !strings.Contains(out.String(), folder+": protected") {
		t.Fatalf("stdout %q", out.String())
	}
	if err := privatedir.CheckModelFolder(folder); err != nil {
		t.Fatal(err)
	}
	// The check walks every folder from the volume's root. A real models
	// root sits in a home that only this account and the machine can
	// change; this test's root sits in the temporary folder, which other
	// principals may reach. Whatever the check refuses must therefore lie
	// above the set folder: the protected folder and its files pass.
	err = trustedpath.CheckFile(filepath.Join(folder, set.Files[0].Name), owners)
	if err != nil {
		msg := err.Error()
		i := strings.Index(msg, ": folder ")
		if i < 0 {
			t.Fatalf("the protected torch file itself fails the trusted-in-place check: %v", err)
		}
		named, _, _ := strings.Cut(msg[i+len(": folder "):], ": ")
		if strings.HasPrefix(strings.ToLower(named), strings.ToLower(folder)) {
			t.Fatalf("the protected set folder fails the trusted-in-place check: %v", err)
		}
		t.Logf("refused above the set folder, as a temporary folder is: %v", err)
	}
}
