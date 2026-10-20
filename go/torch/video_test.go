package torch

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// torchVenv is a venv of a base CPython made once per test binary, without
// pip: the stand-in engine needs the standard library alone. On Windows it
// is the test component's (torchComponent), which the engine runs in its
// sandbox. Its folder is removed by TestMain.
var torchVenv = sync.OnceValues(func() (string, error) {
	if c, err := torchComponent(); err != nil || c != nil {
		if err != nil {
			return "", err
		}
		return c.Python(), nil
	}
	base, err := basePython()
	if err != nil {
		return "", err
	}
	dir, err := os.MkdirTemp("", "pytorch-torch-")
	if err != nil {
		return "", err
	}
	torchVenvDir = dir
	venv := filepath.Join(dir, "venv")
	args := []string{"-m", "venv", "--without-pip"}
	if runtime.GOOS != "windows" {
		// A copied interpreter, since a placed file is never a link.
		args = append(args, "--copies")
	}
	if out, err := exec.Command(base, append(args, venv)...).CombinedOutput(); err != nil {
		return "", errors.New("python -m venv: " + err.Error() + ": " + string(out))
	}
	if runtime.GOOS == "windows" {
		return filepath.Join(venv, "Scripts", "python.exe"), nil
	}
	return filepath.Join(venv, "bin", "python"), nil
})

var torchVenvDir string

func TestMain(m *testing.M) {
	code := m.Run()
	if torchVenvDir != "" {
		//unchecked: a temporary venv a later run does not read
		os.RemoveAll(torchVenvDir)
	}
	// The test component's sandbox must leave nothing: no grant, no access
	// list changed, no profile (KIT-S4).
	if err := removeTestComponent(); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: the test component: %v\n", err)
		code = 1
	}
	os.Exit(code)
}

// basePython is a CPython 3.10 or later on this machine: the py launcher's
// on Windows, python3 elsewhere.
func basePython() (string, error) {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		if _, err := exec.LookPath("py"); err != nil {
			return "", errors.New("no py launcher")
		}
		cmd = exec.Command("py", "-3", "-c", "import sys; assert sys.version_info >= (3, 10); print(sys.executable)")
	} else {
		cmd = exec.Command("python3", "-c", "import sys; assert sys.version_info >= (3, 10); print(sys.executable)")
	}
	out, err := cmd.Output()
	if err != nil {
		return "", errors.New("no CPython 3.10 or later: " + err.Error())
	}
	return strings.TrimSpace(string(out)), nil
}

// testTorchRuntime pins the test venv's interpreter as the reviewed one.
func testTorchRuntime(t *testing.T, python string) torchRuntime {
	t.Helper()
	cfg, err := os.ReadFile(filepath.Join(filepath.Dir(filepath.Dir(python)), "pyvenv.cfg"))
	if err != nil {
		t.Fatal(err)
	}
	version, home := "", ""
	scanner := bufio.NewScanner(bytes.NewReader(cfg))
	for scanner.Scan() {
		key, value, _ := strings.Cut(scanner.Text(), "=")
		switch strings.TrimSpace(key) {
		case "version", "version_info":
			version = strings.TrimSpace(value)
		case "home":
			home = strings.TrimSpace(value)
		}
	}
	pin := func(p string) enginefiles.File {
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		return enginefiles.File{Path: filepath.Base(p), Size: int64(len(data)), SHA256: hex.EncodeToString(sum[:])}
	}
	// The base interpreter's files that run, as the reviewed pin names them.
	var base []enginefiles.File
	for _, pattern := range []string{"python.exe", "python3*.dll", "vcruntime140*.dll", "python3", "DLLs/_socket.pyd", "DLLs/select.pyd"} {
		matches, _ := filepath.Glob(filepath.Join(home, filepath.FromSlash(pattern)))
		for _, m := range matches {
			if info, err := os.Lstat(m); err == nil && info.Mode().IsRegular() {
				f := pin(m)
				rel, _ := filepath.Rel(home, m)
				f.Path = filepath.ToSlash(rel)
				base = append(base, f)
			}
		}
	}
	venv := filepath.Dir(filepath.Dir(python))
	site := "Lib/site-packages"
	if runtime.GOOS != "windows" {
		matches, _ := filepath.Glob(filepath.Join(venv, "lib", "python3*", "site-packages"))
		if len(matches) != 1 {
			t.Fatalf("the test venv's site-packages: %v", matches)
		}
		rel, _ := filepath.Rel(venv, matches[0])
		site = filepath.ToSlash(rel)
	}
	return torchRuntime{Release: "test", Python: pin(python), Base: base, PythonVersion: version, SitePackages: site,
		TargetFamily: "gfx-test"}
}

const torchTestModel = "hummingbird-xt-i2v"

// torchFixture is one placed PyTorch set of small files, AMD's source as two
// small modules, the stand-in engine and the provider's content store.
type torchFixture struct {
	dir    string
	engine TorchEngine
	store  *storage.ContentStore
	record string
	// downloads is where the set's model files lie: outside every grant of
	// the engine's sandbox, so a contained engine reads them only through
	// the handles the provider lends.
	downloads string
}

func newTorchFixture(t *testing.T, args ...string) *torchFixture {
	t.Helper()
	python, err := torchVenv()
	if err != nil {
		t.Skipf("no Python for the stand-in engine: %v", err)
	}
	component, err := torchComponent()
	if err != nil {
		t.Fatalf("the test component: %v", err)
	}
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	modelRoot := root
	manifest := ""
	if component != nil {
		// The engine's own folders lie in the component's granted work
		// folder; the model files stay in the test's own.
		if root, err = os.MkdirTemp(component.work, "fixture-"); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			//unchecked: a fixture folder no later test reads
			os.RemoveAll(root)
		})
		manifest = component.Manifest()
	}
	set := HummingbirdXTI2VTorch
	set.Files, set.Source = nil, nil
	set.Lease = 3 << 30
	downloads := filepath.Join(modelRoot, "downloads")
	for _, f := range HummingbirdXTI2VTorch.Files {
		data := []byte("weights of " + f.Role)
		target := filepath.Join(downloads, f.Name)
		var locators []string
		if strings.HasPrefix(f.Role, "tokenizer") {
			// The tokenizer's four files lie in one folder.
			target = filepath.Join(downloads, "umt5-xxl", f.Name)
			locators = []string{"umt5-xxl/" + f.Name}
		}
		write(t, target, data)
		sum := sha256.Sum256(data)
		set.Files = append(set.Files, VideoModelFile{Name: f.Name, Role: f.Role, Size: int64(len(data)),
			SHA256: hex.EncodeToString(sum[:]), License: "test", Locators: locators})
	}
	source := filepath.Join(root, "source")
	for _, s := range HummingbirdXTI2VTorch.Source[:2] {
		data := []byte("# " + s.Module + "\n")
		write(t, filepath.Join(source, filepath.FromSlash(s.Path)), data)
		sum := sha256.Sum256(data)
		set.Source = append(set.Source, TorchSourceFile{Module: s.Module, Path: s.Path, Size: int64(len(data)), SHA256: hex.EncodeToString(sum[:])})
	}
	state := filepath.Join(root, "state")
	for _, d := range []string{"cache", "temp", "work"} {
		if err := os.MkdirAll(filepath.Join(state, "torch", d), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := PlaceTorchModels(context.Background(), set, []ModelSearch{FolderSearch(downloads)}, TorchModelPlaces(state, set)); err != nil {
		t.Fatal(err)
	}
	realScript := filepath.Join(root, "real", "oa_torch_engine.py")
	write(t, realScript, torchEngineScript)
	fake, err := os.ReadFile(filepath.Join("testdata", "faketorch", "fake_engine.py"))
	if err != nil {
		t.Fatal(err)
	}
	store, err := storage.NewContentStore("video", filepath.Join(root, "content"))
	if err != nil {
		t.Fatal(err)
	}
	record := filepath.Join(root, "record.jsonl")
	return &torchFixture{dir: root, store: store, record: record, downloads: modelRoot, engine: TorchEngine{
		Python: python, Source: source, Set: set, Places: TorchModelPlaces(state, set), ModelRecord: TorchModelRecord(state, set),
		State: filepath.Join(state, "torch"), Files: enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}}),
		Ready: 30 * time.Second, Log: filepath.Join(state, "logs", "torch.log"), Sandbox: manifest,
		runtime: testTorchRuntime(t, python), script: fake,
		fixture: append([]string{"--real=" + realScript, "--record=" + record}, args...),
	}}
}

func write(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func (f *torchFixture) options(leases *fakeLeases) Options {
	engine := f.engine
	return Options{Leases: leases, Idle: time.Hour, Renew: 50 * time.Millisecond, Wait: time.Second,
		Video: &VideoOptions{Torch: &engine, Store: f.store}}
}

func (f *torchFixture) host(t *testing.T, leases *fakeLeases, adjust func(*Options)) *Host {
	t.Helper()
	options := f.options(leases)
	if adjust != nil {
		adjust(&options)
	}
	h, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	if k, ok := h.video.kind.(*torchKind); ok {
		// The hardware encoder runs only under supervision (owner rule on
		// GPU work); the tests take Microsoft's software encoder.
		k.softwareOnly = true
	}
	return h
}

// request is an image-to-video request of one second at the set's 16 fps
// (17 frames) from a staged red first frame.
func (f *torchFixture) request(t *testing.T, id string) wire.VideoRequest {
	t.Helper()
	src := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i], src.Pix[i+3] = 255, 255
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	sum := sha256.Sum256(data)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if _, ok := f.store.Find(digest); !ok {
		ref, err := f.store.Place(digest, int64(len(data)))
		if err != nil {
			t.Fatal(err)
		}
		write(t, f.store.Path(ref), data)
		if err := f.store.Commit(ref); err != nil {
			t.Fatal(err)
		}
	}
	return wire.VideoRequest{RequestID: id, Model: torchTestModel, Prompt: "waves against the lighthouse", Size: "auto",
		DurationMs: 1000, ImageDigest: digest, ImageMediaType: "image/png"}
}

// records are the stand-in engine's records of its loads and jobs.
func (f *torchFixture) records(t *testing.T) []map[string]any {
	t.Helper()
	raw, err := os.ReadFile(f.record)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatal(err)
		}
		out = append(out, m)
	}
	return out
}

func (f *torchFixture) log() string { return readFile(filepath.Dir(f.engine.Log), "torch.log") }

// awaitLog reports whether the engine log holds text within the wait.
func (f *torchFixture) awaitLog(text string, within time.Duration) bool {
	for deadline := time.Now().Add(within); ; time.Sleep(20 * time.Millisecond) {
		if strings.Contains(f.log(), text) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
	}
}

// PYTORCH-T1..T7: the first video loads the engine under the set's
// measured lease, sends the held paths, the source, the dispatch row's fixes
// and the preset, follows progress and keeps the provider's own H.264 MP4 of
// the engine's frames, its reason naming the encoder that ran.
func TestTorchVideoGeneratesThroughTheEngine(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	request := f.request(t, "job-1")
	request.Seed = 1 << 40
	op := startVideo(t, h, request)
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateCompleted || observed.Delivery == nil {
		t.Fatalf("observed %+v; log %s", observed, f.log())
	}
	d := observed.Delivery
	ref, ok := f.store.Find(d.Digest)
	if !ok || d.MediaType != "video/mp4" || d.Location != "local" || d.Size != ref.Size {
		t.Fatalf("delivery %+v", d)
	}
	if observed.Reason != videoEncoderSoftware {
		t.Fatalf("reason %q, want the software encoder the tests ask for", observed.Reason)
	}
	data, err := os.ReadFile(f.store.Path(ref))
	if err != nil || http.DetectContentType(data) != "video/mp4" {
		t.Fatalf("stored result: %v", err)
	}
	if samples := mp4Samples(t, data); samples != 17 {
		t.Fatalf("%d samples", samples)
	}
	if leases.grants[0] != f.engine.Set.Lease || len(leases.grants) != 1 {
		t.Fatalf("lease amounts %v, want the set's %d", leases.grants, f.engine.Set.Lease)
	}
	records := f.records(t)
	if len(records) != 2 || records[0]["event"] != "load" || records[1]["event"] != "generate" {
		t.Fatalf("records %v", records)
	}
	load := records[0]["load"].(map[string]any)
	files := load["files"].(map[string]any)
	for _, file := range f.engine.Set.Files {
		entry := files[file.Role].(map[string]any)
		if entry["sha256"] != file.SHA256 {
			t.Fatalf("%s: %v", file.Role, entry)
		}
		if f.engine.Sandbox != "" {
			if uri, ok := entry["uri"].(string); !ok || !strings.HasPrefix(uri, "fd:") || entry["path"] != nil {
				t.Fatalf("%s was not lent by handle: %v", file.Role, entry)
			}
		} else if p, ok := entry["path"].(string); !ok || !filepath.IsAbs(p) || !strings.EqualFold(filepath.Base(p), file.Name) {
			t.Fatalf("%s: %v", file.Role, entry)
		}
	}
	if fixes := load["fixes"].([]any); len(fixes) != 2 || fixes[0] != "cudnn-off" || fixes[1] != "seq-len-from-clip" {
		t.Fatalf("fixes %v", fixes)
	}
	if opts := load["options"].(map[string]any); opts["decode_dtype"] != "f32" {
		t.Fatalf("the BF16 decode is opt-in: %v", opts)
	}
	if len(load["source"].(map[string]any)) != 2 || len(load["tables"].(map[string]any)) != 2 {
		t.Fatalf("source %v tables %v", load["source"], load["tables"])
	}
	preset := load["preset"].(map[string]any)
	if preset["shift"] != float64(10) || len(preset["timesteps"].([]any)) != 3 {
		t.Fatalf("preset %v", preset)
	}
	job := records[1]
	if job["frames"] != float64(17) || job["width"] != float64(512) || job["fps"] != float64(16) || job["seed"] != float64(1<<40) {
		t.Fatalf("job %v", job)
	}
	// The first frame reached the engine as the provider's own RGB of the
	// set's size: red stays red.
	if px := job["first_pixel"].([]any); px[0] != float64(255) || px[1] != float64(0) {
		t.Fatalf("first pixel %v", px)
	}
	if flags := records[0]["flags"].(map[string]any); flags["isolated"] != float64(1) || flags["dont_write_bytecode"] != float64(1) || flags["no_site"] != float64(1) {
		t.Fatalf("the interpreter runs without -I -S -B: %v", flags)
	}
	site := filepath.Join(filepath.Dir(filepath.Dir(f.engine.Python)), filepath.FromSlash(f.engine.runtime.SitePackages))
	if got, _ := load["site_packages"].(string); !strings.EqualFold(filepath.Clean(got), site) {
		t.Fatalf("site_packages %q, want the venv's %s", got, site)
	}
	if path := records[0]["path"].([]any); slices.ContainsFunc(path, func(p any) bool { return strings.Contains(strings.ToLower(p.(string)), "site-packages") }) {
		t.Fatalf("a site-packages folder was on the path before the load added it: %v", path)
	}
	if after := records[0]["path_after"].([]any); len(after) == 0 || after[len(after)-1] != load["site_packages"] {
		t.Fatalf("the venv's site-packages is not last on the path after the load: %v", after)
	}
	env := records[0]["env"].(map[string]any)
	row, _ := gpufault.DispatchFor(gpufault.EnginePyTorchROCm)
	for name, value := range row.Env {
		if env[name] != value {
			t.Fatalf("%s=%v, want the dispatch row's %s", name, env[name], value)
		}
	}
	for name := range env {
		upper := strings.ToUpper(name)
		if strings.HasPrefix(upper, "PYTHON") || strings.HasPrefix(upper, "GO") || upper == "HOME" && runtime.GOOS == "windows" {
			t.Fatalf("the engine inherited %s", name)
		}
	}
	if env["HF_HUB_OFFLINE"] != "1" || env["ROCM_SDK_TARGET_FAMILY"] != "gfx-test" ||
		!strings.HasPrefix(env["PATH"].(string), filepath.Dir(f.engine.Python)) {
		t.Fatalf("env %v", env)
	}
	if cwd := records[0]["cwd"].(string); !strings.EqualFold(filepath.Clean(cwd), filepath.Join(f.engine.State, "work")) {
		t.Fatalf("cwd %s", cwd)
	}
}

// mp4Samples is the sample count of an MP4's one video track, read from its
// sample size table, after checking that moov comes before mdat.
func mp4Samples(t *testing.T, data []byte) int {
	t.Helper()
	type box struct {
		kind string
		body []byte
	}
	boxes := func(data []byte) []box {
		var out []box
		for len(data) >= 8 {
			size := int(binary.BigEndian.Uint32(data))
			if size < 8 || size > len(data) {
				t.Fatalf("a box of %d bytes in %d", size, len(data))
			}
			out = append(out, box{string(data[4:8]), data[8:size]})
			data = data[size:]
		}
		return out
	}
	child := func(parent []byte, kind string) []byte {
		for _, b := range boxes(parent) {
			if b.kind == kind {
				return b.body
			}
		}
		t.Fatalf("no %s box", kind)
		return nil
	}
	var kinds []string
	var moov []byte
	for _, b := range boxes(data) {
		kinds = append(kinds, b.kind)
		if b.kind == "moov" {
			moov = b.body
		}
	}
	if slices.Index(kinds, "moov") < 0 || slices.Index(kinds, "moov") > slices.Index(kinds, "mdat") {
		t.Fatalf("top-level boxes %v", kinds)
	}
	stsz := child(child(child(child(child(moov, "trak"), "mdia"), "minf"), "stbl"), "stsz")
	return int(binary.BigEndian.Uint32(stsz[8:]))
}

// PYTORCH-T6: progress comes per step and per decode chunk as protocol
// messages, steps filling the set's share for the decode precision.
func TestTorchVideoProgressPerStepAndDecode(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--steps=4", "--step=0.4")
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, f.request(t, "job-1"))
	seen := map[int64]bool{}
	deadline := time.Now().Add(60 * time.Second)
	for {
		o := h.VideoServer().Observe(context.Background(), testCaller, op)
		if o.Status == wire.VideoStateGenerating && o.Step > 0 {
			if o.Steps != 100 || o.Step > 99 {
				t.Fatalf("progress %d of %d", o.Step, o.Steps)
			}
			seen[o.Step] = true
		}
		if o.Status == wire.VideoStateCompleted || o.Status == wire.VideoStateFailed {
			if o.Status != wire.VideoStateCompleted {
				t.Fatalf("observed %+v", o)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the video did not end")
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Four steps of the F32 share, 48: 12, 24, 36 and 48.
	if !seen[12] || !seen[24] || !seen[36] {
		t.Fatalf("progress seen %v", seen)
	}
}

// PYTORCH-T6: a cancel reaches the engine between two steps; the engine
// stays resident under its lease and serves the next video without a load.
func TestTorchVideoCancelBetweenStepsKeepsTheEngine(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--steps=40", "--step=0.05")
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	op := startVideo(t, h, f.request(t, "job-1"))
	waitUntil(t, func() bool {
		o := h.VideoServer().Observe(context.Background(), testCaller, op)
		return o.Status == wire.VideoStateGenerating && o.Step > 0
	})
	if c := h.VideoServer().Cancel(context.Background(), testCaller, op); c.Outcome != wire.CancelOutcomeCancelled {
		t.Fatalf("cancel %+v", c)
	}
	waitUntil(t, func() bool {
		for _, r := range f.records(t) {
			if r["event"] == "after_cancel" {
				return true
			}
		}
		return false
	})
	if !h.Loaded(torchTestModel) {
		t.Fatal("a cancel between steps stopped the engine")
	}
	second := startVideo(t, h, f.request(t, "job-2"))
	if o := awaitVideo(t, h, second, wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("second video %+v", o)
	}
	if acquires, released, _ := leases.counts(); acquires != 1 || len(released) != 0 {
		t.Fatalf("acquires %d released %v", acquires, released)
	}
}

// PYTORCH-T6: an engine that does not stop within the cancel's bound is
// stopped, and its lease released.
func TestTorchVideoUnansweredCancelStopsTheEngine(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--steps=1", "--step=60")
	leases := newFakeLeases()
	h := f.host(t, leases, func(o *Options) {})
	h.video.kind.(*torchKind).cancelWithin = 300 * time.Millisecond
	op := startVideo(t, h, f.request(t, "job-1"))
	awaitVideo(t, h, op, wire.VideoStateGenerating)
	h.VideoServer().Cancel(context.Background(), testCaller, op)
	if !awaitUnloaded(h) {
		t.Fatal("the engine kept running past the cancel's bound")
	}
	waitUntil(t, func() bool { _, released, _ := leases.counts(); return len(released) == 1 })
}

// PYTORCH-T5, T6: an engine that crashes mid-video is lost; its lease is
// released once the holder sees it exit.
func TestTorchVideoEngineCrashIsLost(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--mode=crash-generate")
	leases := newFakeLeases()
	exits := &engineExits{}
	h := f.host(t, leases, func(o *Options) { o.OnEngineExit = exits.hook })
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != videoEngineLost {
		t.Fatalf("observed %+v", o)
	}
	if !awaitUnloaded(h) {
		t.Fatal("a crashed engine stayed resident")
	}
	waitUntil(t, func() bool { _, released, _ := leases.counts(); return len(released) == 1 })
	// KIT-P3: the engine's own exit code is recorded, through the
	// venv launcher, and is the engine log's last line.
	x := exits.await(t)
	if x.Engine != "oa-torch-engine" || x.Model != f.engine.Set.Name || x.Code != 7 || x.Stopped || !x.Notable() {
		t.Fatalf("exit record %+v", x)
	}
	waitUntil(t, func() bool { return strings.Contains(f.log(), "pytorch: "+x.Line()) })
	if lines := strings.Split(strings.TrimSpace(f.log()), "\n"); lines[len(lines)-1] != "pytorch: "+x.Line() {
		t.Fatalf("the exit line is not the log's last: %q", lines[len(lines)-1])
	}
}

// KIT-P3: an engine that dies of a native fault leaves its status in
// the exit record, the job's report of the member's abnormal exit, and the
// Python stack of every thread in its log from faulthandler, which the
// engine script enables before any extension module is imported.
func TestTorchVideoEngineFaultIsRecorded(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--mode=fault-generate")
	exits := &engineExits{}
	h := f.host(t, newFakeLeases(), func(o *Options) { o.OnEngineExit = exits.hook })
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != videoEngineLost {
		t.Fatalf("observed %+v", o)
	}
	x := exits.await(t)
	log := f.log()
	if runtime.GOOS == "windows" {
		if x.Status() != "0xC0000005 STATUS_ACCESS_VIOLATION" || len(x.AbnormalExits) == 0 || x.Stopped {
			t.Fatalf("exit record %+v", x)
		}
		if !strings.Contains(log, "Windows fatal exception: access violation") {
			t.Fatalf("no faulthandler trace in the engine log:\n%s", log)
		}
	} else {
		if x.Signal != "segmentation fault" || !x.Crash() {
			t.Fatalf("exit record %+v", x)
		}
		if !strings.Contains(log, "Fatal Python error: Segmentation fault") {
			t.Fatalf("no faulthandler trace in the engine log:\n%s", log)
		}
	}
	if !strings.Contains(log, "in generate") || !strings.Contains(log, "pytorch: "+x.Line()) {
		t.Fatalf("the trace does not name the job's frame, or the exit line is missing:\n%s", log)
	}
	// The session's job line and the set's last phase place the fault.
	records := f.records(t)
	if len(records) < 2 || records[len(records)-1]["event"] != "generate" {
		t.Fatalf("records %v", records)
	}
	id := records[len(records)-1]["id"].(string)
	for _, line := range []string{`"event": "generate", "id": "` + id + `", "set": "hummingbird-xt-i2v"`,
		`"event": "text_encode", "id": "` + id + `", "at": "end"`} {
		if !strings.Contains(log, line) {
			t.Fatalf("no %s in the engine log:\n%s", line, log)
		}
	}
}

// PYTORCH-T6: failures carry a code alone; no engine text reaches the
// caller.
func TestTorchVideoFailureCarriesNoEngineText(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--mode=fail")
	h := f.host(t, newFakeLeases(), nil)
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != videoEngineFailed {
		t.Fatalf("observed %+v", o)
	}
	// The engine writes its traceback before it answers failed, but the
	// provider copies its stderr into the log on its own goroutine, so the
	// line may land just after the observation.
	if !f.awaitLog("never leaves the engine", 10*time.Second) {
		t.Fatalf("the engine's own text is not in its log: %s", f.log())
	}
	if !h.Loaded(torchTestModel) {
		t.Fatal("a failed job stopped the engine")
	}
}

// PYTORCH-T8 with RES-G1..G3: a device-lost line in the engine's own form,
// confirmed by the engine's exit, is a reset: the video fails gpu:reset and
// its fingerprint is quarantined.
func TestTorchVideoDeviceLostConfirmedIsAReset(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--mode=device-lost")
	r := newGPURig(t)
	leases := newFakeLeases()
	h := f.host(t, leases, func(o *Options) { o.GPU = r.guard })
	request := f.request(t, "job-1")
	o := awaitVideo(t, h, startVideo(t, h, request), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != gpufault.WordReset {
		t.Fatalf("observed %+v; log %s", o, f.log())
	}
	fault := nextGPUFault(t, r.faults)
	if fault.Source != "oa-torch-engine: hipErrorLaunchFailure" || len(fault.Quarantined) != 1 ||
		!strings.Contains(fault.Quarantined[0].Fingerprint.Model, "set=torch:"+torchTestModel) {
		t.Fatalf("fault %+v", fault)
	}
	again := request
	again.RequestID = "job-2"
	if a := h.VideoServer().Start(context.Background(), testCaller, again); a.Outcome != wire.StartOutcomeUnavailable || a.Reason != gpufault.WordQuarantined {
		t.Fatalf("the identical request was admitted: %+v", a)
	}
	waitUntil(t, func() bool { _, released, _ := leases.counts(); return len(released) == 1 })
}

// PYTORCH-T8, RES-G1: a device-lost line from an engine that keeps
// answering is not a reset; nothing is stopped or quarantined.
func TestTorchVideoDeviceLostLineWhileAnsweringIsNoReset(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--mode=device-line-alive")
	r := newGPURig(t)
	reported := make(chan error, 8)
	h := f.host(t, newFakeLeases(), func(o *Options) {
		o.GPU = r.guard
		o.OnError = func(err error) { reported <- err }
	})
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateCompleted {
		t.Fatalf("observed %+v", o)
	}
	select {
	case err := <-reported:
		if !strings.Contains(err.Error(), "still answers") {
			t.Fatalf("reported %v", err)
		}
	case <-time.After(deviceLostConfirm + 10*time.Second):
		t.Fatal("the unconfirmed line was not reported")
	}
	if !h.Loaded(torchTestModel) || len(r.guard.Faults()) != 0 {
		t.Fatalf("loaded %v faults %v", h.Loaded(torchTestModel), r.guard.Faults())
	}
}

// PYTORCH-T5, H2: a lease the resource cannot hold is refused before the
// engine starts.
func TestTorchVideoOversizedLeaseIsRefused(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	leases.capacity = f.engine.Set.Lease - 1
	h := f.host(t, leases, nil)
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != Insufficient {
		t.Fatalf("observed %+v", o)
	}
	if records := f.records(t); len(records) != 0 {
		t.Fatalf("an engine started for a refused lease: %v", records)
	}
}

// PYTORCH-T5, H8: an unload stops the engine and releases its lease.
func TestTorchVideoUnloadReleasesTheLease(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted)
	h.torch.mu.Lock()
	running := h.torch.running
	h.torch.mu.Unlock()
	h.Unload(torchTestModel, "test")
	select {
	case <-running.process.exited:
	case <-time.After(30 * time.Second):
		t.Fatal("the engine survived its unload")
	}
	if _, released, _ := leases.counts(); len(released) != 1 || h.Loaded(torchTestModel) {
		t.Fatalf("released %v loaded %v", released, h.Loaded(torchTestModel))
	}
}

// PYTORCH-T5: a load the engine refuses, and an engine that breaks its
// protocol, end the load and give the lease back.
func TestTorchEngineLoadFailuresGiveTheLeaseBack(t *testing.T) {
	engineRuns(t)
	for _, mode := range []string{"error-load", "crash-load", "garbage"} {
		t.Run(mode, func(t *testing.T) {
			f := newTorchFixture(t, "--mode="+mode)
			leases := newFakeLeases()
			var mu sync.Mutex
			var reported []string
			h := f.host(t, leases, func(o *Options) {
				o.OnError = func(err error) { mu.Lock(); reported = append(reported, err.Error()); mu.Unlock() }
			})
			o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
			if o.Status != wire.VideoStateFailed || o.Reason != "engine:unavailable" {
				t.Fatalf("observed %+v", o)
			}
			mu.Lock()
			said := strings.Join(reported, "\n")
			mu.Unlock()
			if mode == "error-load" && !strings.Contains(said, `code files at stage "transformer"`) {
				t.Fatalf("the load's code was not reported: %q", said)
			}
			if acquires, released, _ := leases.counts(); acquires != 1 || len(released) != 1 {
				t.Fatalf("acquires %d released %v", acquires, released)
			}
		})
	}
}

// PYTORCH-T2: a source file that is not the pin is refused before any
// lease.
func TestTorchSourceIsProvedBeforeTheLease(t *testing.T) {
	f := newTorchFixture(t)
	write(t, filepath.Join(f.engine.Source, filepath.FromSlash(f.engine.Set.Source[0].Path)), []byte("# changed\n"))
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != NoEngine {
		t.Fatalf("observed %+v", o)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatalf("%d leases for an unproved engine", acquires)
	}
}

// PYTORCH-T4: the engine environment admits only the allowlist and the
// dispatch row's variables; PYTHON* and proxies are refused by name.
func TestTorchEnvironmentRefusesForbiddenNames(t *testing.T) {
	for _, env := range [][]string{{"PYTHONPATH=x"}, {"HTTPS_PROXY=x"}, {"ROCM_SDK_PRELOAD_LIBRARIES=x"}, {"GGML_X=1"}, {"PATH=a", "path=b"}} {
		if err := checkTorchEnvironment(env); !errors.Is(err, ErrEngineEnvironment) {
			t.Fatalf("%v: %v", env, err)
		}
	}
	if err := checkTorchEnvironment([]string{"MIOPEN_FIND_MODE=FAST", "PATH=x", "HF_HUB_OFFLINE=1"}); err != nil {
		t.Fatal(err)
	}
}

// RES-G5: every fix of the pytorch-rocm dispatch row is one the engine
// applies, and a request at the known-bad token count is refused.
func TestTorchDispatchRowIsApplied(t *testing.T) {
	fixes, err := torchFixes()
	if err != nil || !slices.Equal(fixes, []string{"cudnn-off", "seq-len-from-clip"}) {
		t.Fatalf("fixes %v %v", fixes, err)
	}
	k := &torchKind{sp: TorchSetPlace{Set: HummingbirdXTI2VTorch}}
	if outcome, reason := k.Admit(sdcpp.VideoParams{Prompt: "surf", Width: 1280, Height: 704, Frames: 121}); outcome != wire.StartOutcomeUnavailable || reason != gpufault.WordKnownBad {
		t.Fatalf("1280x704x121: %s %s", outcome, reason)
	}
	if outcome, _ := k.Admit(sdcpp.VideoParams{Prompt: "surf", Width: 512, Height: 512, Frames: 81}); outcome != wire.StartOutcomeAccepted {
		t.Fatalf("512x512x81: %s", outcome)
	}
}

// PYTORCH-T6: the sd path's own prompt bound holds for the PyTorch
// engine: a line break, another control character, U+2028 or U+2029 is
// refused at Start and no engine starts.
func TestTorchVideoPromptWithALineBreakIsRefused(t *testing.T) {
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	for i, prompt := range []string{"waves\nfake log line", "bell\a", "a b", "a b"} {
		request := f.request(t, fmt.Sprintf("job-%d", i))
		request.Prompt = prompt
		a := h.VideoServer().Start(context.Background(), testCaller, request)
		if a.Outcome != wire.StartOutcomeUnsupportedFeature || a.Reason != sdcpp.ReasonVideoBound {
			t.Fatalf("%q: %+v", prompt, a)
		}
	}
	if acquires, _, _ := leases.counts(); acquires != 0 || len(f.records(t)) != 0 {
		t.Fatalf("an engine started for a refused prompt: %d leases", acquires)
	}
}

// PYTORCH-T3: a header line that never ends is refused once it passes the
// bound, without waiting for its end.
func TestTorchProtocolRefusesAnUnterminatedLongHeader(t *testing.T) {
	r := bufio.NewReader(io.MultiReader(strings.NewReader(`{"type":"loading","stage":"`+strings.Repeat("x", 2*torchMaxHeader)), neverEnding{}))
	if _, err := readTorchMessage(r, 0); !errors.Is(err, errTorchProtocol) {
		t.Fatalf("%v", err)
	}
}

// neverEnding is an engine output that blocks forever after its first
// bytes; a read past the header's bound must not reach it.
type neverEnding struct{}

func (neverEnding) Read([]byte) (int, error) { select {} }

// PYTORCH-T4, INF-EL3: the path-length gate runs on the venv's hipBLASLt
// and rocBLAS kernel folders before the lease; a folder too deep for its
// kernel paths with long paths off fails the video engine:gpu-path, and
// with long paths on the video runs.
func TestTorchGPUPathGateRunsBeforeTheLease(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	// The gate reads the libraries below the venv: this test's own folder
	// there, with kernel paths over MAX_PATH.
	venv := filepath.Dir(filepath.Dir(f.engine.Python))
	libs := filepath.Join(venv, "oa-test-libs")
	t.Cleanup(func() { _ = os.RemoveAll(libs) })
	for _, lib := range []string{"hipblaslt", "rocblas"} {
		dir := filepath.Join(libs, lib, "library", "gfx-test")
		write(t, filepath.Join(dir, "TensileLibrary_"+strings.Repeat("x", 200)+".dat"), []byte("k"))
		if enginelaunch.UTF16Len(dir)+220 <= enginelaunch.MaxLibraryPath {
			t.Skip("the temporary folder is too short for a kernel path over MAX_PATH")
		}
	}
	f.engine.runtime.Libraries = "oa-test-libs"
	off := enginelaunch.GPUProbe{
		LongPathsEnabled: func() bool { return false },
		ShortPath:        func(string) (string, error) { return "", errors.New("no 8.3 names") },
		FinalPath:        func(p string) (string, error) { return p, nil },
		ReparsePoint:     func(string) (bool, error) { return false, nil },
	}
	f.engine.gpuProbe = &off
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != errGPUPath {
		t.Fatalf("observed %+v", o)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatalf("%d leases for a gated launch", acquires)
	}
	on := off
	on.LongPathsEnabled = func() bool { return true }
	f.engine.gpuProbe = &on
	h = f.host(t, newFakeLeases(), nil)
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-2")), wire.VideoStateFailed, wire.VideoStateCompleted); o.Status != wire.VideoStateCompleted {
		t.Fatalf("long paths on: %+v", o)
	}
}

// PYTORCH-T1: the base interpreter pyvenv.cfg names is proved with the
// launcher; a file of it that is not the pin refuses the video before any
// lease.
func TestTorchBaseInterpreterIsProved(t *testing.T) {
	f := newTorchFixture(t)
	if len(f.engine.runtime.Base) == 0 {
		t.Skip("the test venv's base interpreter has no regular file to pin")
	}
	f.engine.runtime.Base = append([]enginefiles.File(nil), f.engine.runtime.Base...)
	f.engine.runtime.Base[0].SHA256 = strings.Repeat("0", 64)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if o.Status != wire.VideoStateFailed || o.Reason != NoEngine {
		t.Fatalf("observed %+v", o)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatalf("%d leases for an unproved interpreter", acquires)
	}
}

// PYTORCH-T2, T5: the reviewed row is valid, its lease is the measured
// 31.0 GiB and its BF16 decode is an option, F32 by default.
func TestHummingbirdTorchRowIsValid(t *testing.T) {
	s := HummingbirdXTI2VTorch
	if err := s.valid(); err != nil {
		t.Fatal(err)
	}
	if s.Lease != 33285996544 || s.option(nil, "decode-dtype") != "f32" || s.option(map[string]string{"decode-dtype": "bf16"}, "decode-dtype") != "bf16" {
		t.Fatalf("lease %d default %s", s.Lease, s.option(nil, "decode-dtype"))
	}
	if err := s.checkOptions(map[string]string{"decode-dtype": "f16"}); err == nil {
		t.Fatal("an unknown decode precision passed")
	}
	if len(s.Source) != 8 || s.FirstFrame != FirstFrameRequired || s.UsePolicy.Name == "" {
		t.Fatalf("row %+v", s)
	}
	// PYTORCH-T1: the native modules the network hook's import of socket
	// loads (_socket, and select through selectors) are pinned with the
	// base interpreter.
	for _, name := range []string{"DLLs/_socket.pyd", "DLLs/select.pyd"} {
		if !slices.ContainsFunc(reviewedTorchRuntime.Base, func(f enginefiles.File) bool { return f.Path == name }) {
			t.Fatalf("the reviewed runtime does not pin %s: %+v", name, reviewedTorchRuntime.Base)
		}
	}
}

// PYTORCH-T3: the provider refuses every engine message outside the
// protocol's closed list before acting on it.
func TestTorchProtocolRefusesMalformedMessages(t *testing.T) {
	read := func(text string, frame int64) error {
		_, err := readTorchMessage(bufio.NewReader(strings.NewReader(text)), frame)
		return err
	}
	for name, text := range map[string]string{
		"not json":       "hello\n",
		"unknown type":   `{"type":"shell"}` + "\n",
		"extra field":    `{"type":"ready","text":"x"}` + "\n",
		"unknown code":   `{"type":"failed","id":"a","code":"Traceback"}` + "\n",
		"bad progress":   `{"type":"progress","id":"a","phase":"step","done":4,"total":3}` + "\n",
		"payload":        `{"type":"ready","bytes":4}` + "\n",
		"frame size":     `{"type":"frame","id":"a","index":0,"bytes":5}` + "\n12345",
		"short frame":    `{"type":"frame","id":"a","index":0,"bytes":12}` + "\n1234",
		"pong no health": `{"type":"pong","seq":1}` + "\n",
		"hello version":  `{"type":"hello","protocol":1,"pid":4}` + "\n",
		"ready no set":   `{"type":"ready"}` + "\n",
		"error no set":   `{"type":"error","code":"load"}` + "\n",
		"long set":       `{"type":"unloaded","set":"` + strings.Repeat("x", 65) + `"}` + "\n",
		"long header":    `{"type":"loading","stage":"` + strings.Repeat("x", torchMaxHeader) + `"}` + "\n",
	} {
		if err := read(text, 12); !errors.Is(err, errTorchProtocol) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	m, err := readTorchMessage(bufio.NewReader(strings.NewReader(`{"type":"frame","id":"a","index":0,"bytes":3}`+"\nabc")), 3)
	if err != nil || string(m.payload) != "abc" {
		t.Fatalf("frame %v %q", err, m.payload)
	}
}

// The first frame reaches the engine as raw RGB of the request's size.
func TestTorchFirstFrameIsRawRGB(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	src.Set(0, 0, color.NRGBA{10, 20, 30, 255})
	src.Set(1, 0, color.NRGBA{40, 50, 60, 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	out, err := rgb(buf.Bytes(), 2, 1)
	if err != nil || !bytes.Equal(out, []byte{10, 20, 30, 40, 50, 60}) {
		t.Fatalf("%v %v", out, err)
	}
	if _, err := rgb(buf.Bytes(), 4, 4); err == nil {
		t.Fatal("a frame of another size passed")
	}
}
