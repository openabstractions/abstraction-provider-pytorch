//go:build windows

package torch

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-download/go/archive"
	"github.com/openabstractions/abstraction-provider-kit/go/modelbytes"
	"github.com/openabstractions/abstraction-provider-kit/go/modelfiles"
	storage "github.com/openabstractions/abstraction-storage/go"
	"golang.org/x/sys/windows"
)

// TestMeasureModelLoad times a model set's files reaching the GPU one after
// another through the engine script's own loader (model_bytes) on a real
// PyTorch stack, each lent when the load asks for it, as pytorchd lends
// them (KIT-MB4). It runs only when OA_TORCHLOAD_FILES is set; it is one
// short GPU run, for the owner's machine.
//
//	OA_TORCHLOAD_FILES    PATH=SHA256 entries separated by ";", in load order
//	OA_TORCHLOAD_MODE     file (a lent file handle) or section (OA fills,
//	                      proves and lends a sealed section)
//	OA_TORCHLOAD_COLD     1 drops the files from the file cache first
//	OA_TORCHLOAD_RECORDS  a span record folder kept between runs (section)
//	OA_TORCHLOAD_PYTHON   the interpreter with torch on ROCm; default py -3.12
//	OA_TORCHLOAD_SCRIPT   an engine script to measure in place of this one
//	OA_TORCHLOAD_SOURCE   archive fills each section from the person's download
//	                      archive in place of the local file, which then only
//	                      names the size; OA_TORCHLOAD_ARCHIVE_STATE names the
//	                      folder of its setting (%APPDATA%\OpenAbstractions\setup)
func TestMeasureModelLoad(t *testing.T) {
	spec := os.Getenv("OA_TORCHLOAD_FILES")
	if spec == "" {
		t.Skip("OA_TORCHLOAD_FILES is not set")
	}
	mode, source := os.Getenv("OA_TORCHLOAD_MODE"), os.Getenv("OA_TORCHLOAD_SOURCE")
	if source == "archive" {
		// The tests run with the per-user folders moved aside
		// (personfolders), so the person's archive setting is named.
		dir := os.Getenv("OA_TORCHLOAD_ARCHIVE_STATE")
		if dir == "" {
			t.Fatal("OA_TORCHLOAD_ARCHIVE_STATE names the folder holding download-archive.json")
		}
		archive.UseState(dir)
		defer archive.UseState("")
	}
	type file struct {
		path, sha string
		size      int64
	}
	var files []file
	for _, part := range strings.Split(spec, ";") {
		path, sha, ok := strings.Cut(part, "=")
		st, err := os.Stat(path)
		if !ok || err != nil {
			t.Fatalf("OA_TORCHLOAD_FILES entry %q: %v", part, err)
		}
		files = append(files, file{path, sha, st.Size()})
	}
	python := os.Getenv("OA_TORCHLOAD_PYTHON")
	if python == "" {
		out, err := exec.Command("py", "-3.12", "-c", "import sys; print(sys.executable)").Output()
		if err != nil {
			t.Fatal(err)
		}
		python = strings.TrimSpace(string(out))
	}
	dir := t.TempDir()
	script, receiver := filepath.Join(dir, "oa_torch_engine.py"), filepath.Join(dir, "oa_model_bytes.py")
	engineScript := torchEngineScript
	if other := os.Getenv("OA_TORCHLOAD_SCRIPT"); other != "" {
		var err error
		if engineScript, err = os.ReadFile(other); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(script, engineScript, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(receiver, modelbytes.PythonReceiver, 0o600); err != nil {
		t.Fatal(err)
	}
	m := &ModelMemory{}
	var records *modelfiles.SpanRecords
	if d := os.Getenv("OA_TORCHLOAD_RECORDS"); d != "" {
		var err error
		if records, err = modelfiles.OpenSpanRecords(d); err != nil {
			t.Fatal(err)
		}
	}
	m.Records = records
	if os.Getenv("OA_TORCHLOAD_COLD") == "1" {
		for _, f := range files {
			purge(t, f.path)
		}
	}
	child, err := filepath.Abs(filepath.Join("testdata", "loadmeasure", "child.py"))
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"-I", "-B", child, script, receiver}
	for _, f := range files {
		args = append(args, fmt.Sprintf("%d:%s", f.size, f.sha))
	}
	cmd := exec.Command(python, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	process, err := windows.OpenProcess(windows.PROCESS_DUP_HANDLE, false, uint32(cmd.Process.Pid))
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(process)
	dup := func(h uintptr, access uint32) (uintptr, error) {
		var target windows.Handle
		err := windows.DuplicateHandle(windows.CurrentProcess(), windows.Handle(h), process, &target, access, false, 0)
		return uintptr(target), err
	}
	// live and peak are the bytes of sections held now and at most.
	var mu sync.Mutex
	var live, peak int64
	var fills []float64
	var result map[string]any
	sc := bufio.NewScanner(stdout)
	for sc.Scan() {
		line := sc.Text()
		var i int
		if _, err := fmt.Sscanf(line, "need %d", &i); err != nil {
			if err := json.Unmarshal([]byte(line), &result); err != nil {
				t.Fatalf("child: %q", line)
			}
			continue
		}
		f := files[i]
		var arg string
		switch mode {
		case "file":
			h, err := os.Open(f.path)
			if err != nil {
				t.Fatal(err)
			}
			v, err := dup(h.Fd(), 0x120089)
			h.Close()
			if err != nil {
				t.Fatal(err)
			}
			arg = modelbytes.URI{Handle: uint64(v), Length: uint64(f.size), SHA256: f.sha}.String()
		case "section":
			var began time.Time
			m.Hold = func(n int64) (func(), error) {
				began = time.Now()
				mu.Lock()
				live += n
				peak = max(peak, live)
				mu.Unlock()
				return func() { mu.Lock(); live -= n; mu.Unlock() }, nil
			}
			var store storage.Store = heldFile{path: f.path, digest: "sha256:" + f.sha, size: f.size}
			if source == "archive" {
				store = archive.Store{}
			}
			ref, ok := store.Find("sha256:" + f.sha)
			if !ok {
				t.Fatalf("the %s store does not find %s", store.Name(), f.sha)
			}
			uri, err := m.lendTo(VideoModelFile{Name: filepath.Base(f.path), Size: f.size, SHA256: f.sha}, store, ref, dup, nil)
			if err != nil {
				t.Fatal(err)
			}
			fills = append(fills, time.Since(began).Seconds())
			arg = uri
		default:
			t.Fatalf("OA_TORCHLOAD_MODE %q is not file or section", mode)
		}
		if _, err := io.WriteString(stdin, arg+"\n"); err != nil {
			t.Fatal(err)
		}
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("%v\n%s", err, stderr.String())
	}
	if result == nil {
		t.Fatalf("no result\n%s", stderr.String())
	}
	var filled float64
	for _, s := range fills {
		filled += s
	}
	bytes := result["bytes"].(float64)
	result["mode"], result["fill_seconds"], result["peak_section_bytes"] = mode, fills, peak
	if filled > 0 {
		result["fill_gbps"] = bytes / filled / 1e9
	}
	result["set_gbps"] = bytes / result["engine_total"].(float64) / 1e9
	line, _ := json.Marshal(result)
	t.Logf("%s", line)
}

// purge drops file from the file cache: an unbuffered open makes Windows
// discard its cached pages when no process maps it (scripts/load_speed.py).
func purge(t *testing.T, file string) {
	t.Helper()
	p, err := windows.UTF16PtrFromString(file)
	if err != nil {
		t.Fatal(err)
	}
	h, err := windows.CreateFile(p, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, windows.FILE_FLAG_NO_BUFFERING, 0)
	if err != nil {
		t.Fatal(err)
	}
	windows.CloseHandle(h)
}
