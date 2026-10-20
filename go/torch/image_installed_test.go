package torch

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// A request for a set whose files are absent is refused unavailable with
// engine:not_installed:pytorch-image before any lease or engine start: a
// set never placed, and a placed set whose file left.
func TestTorchImageRefusesASetWhoseFilesAreAbsent(t *testing.T) {
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.imageHost(t, leases, nil)
	want := "engine:not_installed:" + imageRouteName
	k := h.images.named(torchImageModel)
	where, _, err := readFilePlaces(k.sp.Places, k.sp.Set.Name, k.sp.Set.Files)
	if err != nil || !k.installed() {
		t.Fatalf("the placed fixture reads not installed: %v", err)
	}
	if err := os.Remove(where[k.sp.Set.Files[0].Name]); err != nil {
		t.Fatal(err)
	}
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeUnavailable || reply.reason != want {
		t.Fatalf("a set missing a file ran: %+v", reply)
	}
	if err := os.Remove(k.sp.Places); err != nil {
		t.Fatal(err)
	}
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeUnavailable || reply.reason != want {
		t.Fatalf("a set never placed ran: %+v", reply)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 || len(f.records(t)) != 0 {
		t.Fatalf("an engine started for a set whose files are absent: %d", acquires)
	}
}

// Model verification also runs without the native engine sandbox.
func TestTorchHoldModelsRefusesChangedImageWeights(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	f := torchFixture{downloads: dir, engine: TorchEngine{
		State: filepath.Join(dir, "state", "torch"),
		Files: programfiles.NewVerifier(programfiles.Options{ACL: openACL{}}),
	}}
	sp := f.imageSet(t)
	held, err := f.engine.holdModels(t.Context(), sp, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := held.Release(); err != nil {
		t.Fatal(err)
	}
	changeImageWeights(t, sp)
	if held, err := f.engine.holdModels(t.Context(), sp, false); !errors.Is(err, programfiles.ErrMismatch) {
		if held != nil {
			_ = held.Release()
		}
		t.Fatalf("holding changed weights: %v", err)
	}
}

// A native load that finds a placed file other than the pinned one is refused, and
// the set reads not installed from then on: the next request is refused
// engine:not_installed:pytorch-image before any lease, until models install
// proves the files again.
func TestTorchLoadMismatchMarksTheSetNotInstalled(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.imageHost(t, leases, nil)
	k := h.images.named(torchImageModel)
	file, original := changeImageWeights(t, k.sp)
	first, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse"))
	if first.outcome != wire.ReplyOutcomeUnavailable || first.reason == "engine:not_installed:"+imageRouteName {
		t.Fatalf("the mismatched load = %+v", first)
	}
	if k.installed() {
		t.Fatalf("a set whose load found other bytes reads installed after reply %+v", first)
	}
	second, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse"))
	if second.outcome != wire.ReplyOutcomeUnavailable || second.reason != "engine:not_installed:"+imageRouteName {
		t.Fatalf("the next request = %+v", second)
	}
	// models install puts the pinned bytes back in place and proves them.
	if err := os.WriteFile(file, original, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := PlaceTorchModels(context.Background(), k.sp.Set, []ModelSearch{FolderSearch(filepath.Dir(file))}, k.sp.Places); err != nil {
		t.Fatal(err)
	}
	if err := CheckTorchModels(context.Background(), programfiles.NewVerifier(programfiles.Options{}), k.sp.Set, k.sp.Places, k.sp.ModelRecord); err != nil {
		t.Fatal(err)
	}
	if !k.installed() {
		t.Fatal("a set proved again reads not installed")
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatalf("a lease was taken for a refused set: %d", acquires)
	}
}

func changeImageWeights(t *testing.T, sp TorchSetPlace) (string, []byte) {
	t.Helper()
	where, _, err := readFilePlaces(sp.Places, sp.Set.Name, sp.Set.Files)
	if err != nil {
		t.Fatal(err)
	}
	file := where[sp.Set.Files[0].Name]
	original, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	// Keep the size so the stat-only check passes and the content proof runs.
	if err := os.WriteFile(file, bytes.ToUpper(original), 0o600); err != nil {
		t.Fatal(err)
	}
	return file, original
}
