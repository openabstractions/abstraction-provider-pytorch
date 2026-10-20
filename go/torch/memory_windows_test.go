//go:build windows

package torch

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-provider-kit/go/modelfiles"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// INF-MS9, KIT-MB1..KIT-MB4: with ModelMemory the sandboxed engine receives
// each model file as a sealed memory section OA fills from the held file
// when the engine's load asks for it, with a lease; the engine's real
// receiver maps it inside its AppContainer and the bytes are the pin's; each
// section's holding is taken for the file's size and released once the
// engine has released the section, so the sections are not all held at
// once.
func TestTorchEngineReceivesOAFilledSections(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	if f.engine.Sandbox == "" {
		t.Skip("no sandbox component: sections are lent only to a contained engine")
	}
	records, err := modelfiles.OpenSpanRecords(filepath.Join(t.TempDir(), "spans"))
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var held []int64
	var live, peak int64
	released := 0
	f.engine.Memory = &ModelMemory{Records: records, Hold: func(n int64) (func(), error) {
		mu.Lock()
		defer mu.Unlock()
		held = append(held, n)
		live += n
		peak = max(peak, live)
		return func() {
			mu.Lock()
			released++
			live -= n
			mu.Unlock()
		}, nil
	}}
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, f.request(t, "job-sections"))
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateCompleted {
		t.Fatalf("observed %+v; log %s", observed, f.log())
	}
	records0 := f.records(t)
	files := records0[0]["load"].(map[string]any)["files"].(map[string]any)
	var sizes []int64
	for _, file := range f.engine.Set.Files {
		entry := files[file.Role].(map[string]any)
		if entry["lend"] != "memory" || entry["role"] != file.Role || entry["uri"] != nil || entry["path"] != nil {
			t.Fatalf("%s is not lent on demand: %v", file.Role, entry)
		}
		sizes = append(sizes, file.Size)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		mu.Lock()
		n, got := released, slices.Clone(held)
		mu.Unlock()
		if n == len(sizes) {
			slices.Sort(got)
			slices.Sort(sizes)
			if !slices.Equal(got, sizes) {
				t.Fatalf("memory held %v, want the files' sizes %v", got, sizes)
			}
			var total int64
			for _, n := range sizes {
				total += n
			}
			mu.Lock()
			top := peak
			mu.Unlock()
			if len(sizes) > 2 && top >= total {
				t.Fatalf("every section was held at once: peak %d of %d bytes", top, total)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d of %d sections were freed after the engine loaded them; log %s", n, len(sizes), f.log())
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// archiveStore stands in for the download archive (abstraction-download
// archive.Store): a store over one folder of files named by their hex
// digest, which records the references a reader refused.
type archiveStore struct {
	dir     string
	mu      sync.Mutex
	found   int
	refused []string
}

func (a *archiveStore) Name() string { return "download-archive" }
func (a *archiveStore) Find(digest string) (storage.Ref, bool) {
	p := filepath.Join(a.dir, strings.TrimPrefix(digest, "sha256:"))
	st, err := os.Stat(p)
	if err != nil {
		return storage.Ref{}, false
	}
	a.mu.Lock()
	a.found++
	a.mu.Unlock()
	return storage.NewRef(a.Name(), digest, p, st.Size()), true
}
func (a *archiveStore) Place(string, int64) (storage.Ref, error) {
	return storage.Ref{}, storage.ErrReadOnly
}
func (a *archiveStore) Path(r storage.Ref) string { return storage.Locator(r) }
func (a *archiveStore) Refused(r storage.Ref, _ error) {
	a.mu.Lock()
	a.refused = append(a.refused, r.Digest)
	a.mu.Unlock()
}

// archiveFixture moves the set's first model file from where it was placed
// into an archive stand-in, holding wrong bytes of its size when bad.
func archiveFixture(t *testing.T, bad bool) (*torchFixture, *archiveStore, VideoModelFile) {
	t.Helper()
	f := newTorchFixture(t)
	if f.engine.Sandbox == "" {
		t.Skip("no sandbox component: the archive reaches only a contained engine")
	}
	records, err := modelfiles.OpenSpanRecords(filepath.Join(t.TempDir(), "spans"))
	if err != nil {
		t.Fatal(err)
	}
	a := &archiveStore{dir: t.TempDir()}
	f.engine.Memory = &ModelMemory{Records: records}
	f.engine.Archive = a
	file := f.engine.Set.Files[0]
	placed := loosePlaces(f.engine.Places)[file.Name]
	data, err := os.ReadFile(placed)
	if err != nil {
		t.Fatal(err)
	}
	if bad {
		data[0] ^= 0xff
	}
	if err := os.WriteFile(filepath.Join(a.dir, file.SHA256), data, 0o600); err != nil {
		t.Fatal(err)
	}
	// The local copy is gone; the places record still names it.
	if err := os.Remove(placed); err != nil {
		t.Fatal(err)
	}
	return f, a, file
}

// PYTORCH-T12, storage STO-D2: a model file missing where it was placed is
// filled from the archive, proved and lent, and the engine loads; the other
// files come from where they were placed.
func TestTorchFillsAMissingModelFileFromTheArchive(t *testing.T) {
	engineRuns(t)
	f, a, file := archiveFixture(t, false)
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, f.request(t, "job-archive"))
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateCompleted {
		t.Fatalf("observed %+v; log %s", observed, f.log())
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.found == 0 || len(a.refused) != 0 {
		t.Fatalf("the archive was asked %d times and refused %v", a.found, a.refused)
	}
	entry := f.records(t)[0]["load"].(map[string]any)["files"].(map[string]any)[file.Role].(map[string]any)
	if entry["lend"] != "memory" || entry["path"] != nil {
		t.Fatalf("%s reached the engine as %v", file.Role, entry)
	}
}

// DL-R44: archive bytes that differ from the digest are refused before any
// engine maps them, the archive's bad-bytes rule is applied, and the load
// fails until the file is placed or downloaded.
func TestTorchRefusesDifferingArchiveBytes(t *testing.T) {
	engineRuns(t)
	f, a, file := archiveFixture(t, true)
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, f.request(t, "job-archive-bad"))
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateFailed {
		t.Fatalf("differing archive bytes loaded: %+v", observed)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.refused) != 1 || a.refused[0] != "sha256:"+file.SHA256 {
		t.Fatalf("the archive's rule was applied to %v", a.refused)
	}
}

// R35 review: an archive file of another size than the pin is never used.
// The section is counted at the pinned size, so a larger file would commit
// memory outside the card lease before its digest check.
func TestTorchSkipsAnArchiveFileOfAnotherSize(t *testing.T) {
	engineRuns(t)
	f, a, file := archiveFixture(t, false)
	archived, err := os.OpenFile(filepath.Join(a.dir, file.SHA256), os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := archived.Write(make([]byte, 4096)); err != nil {
		t.Fatal(err)
	}
	archived.Close()
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, f.request(t, "job-archive-size"))
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateFailed {
		t.Fatalf("an archive file of another size loaded: %+v", observed)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.refused) != 0 {
		t.Fatalf("the archive file was filled and refused: %v", a.refused)
	}
}
