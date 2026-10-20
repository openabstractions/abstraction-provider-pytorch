package torch

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openabstractions/abstraction-identity/go/trustedpath"
	"github.com/openabstractions/abstraction-inference/go/providerhost/devicelost"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
	"github.com/openabstractions/abstraction-resource/go/gpufault"

	identity "github.com/openabstractions/abstraction-identity"
)

// The OA PyTorch engine script and the key tables it renames Wan-AI's
// Diffusers tensors by. The provider writes them into its own state, proves
// and holds them there, and the engine runs the held files (PYTORCH-T1).
var (
	//go:embed torch/oa_torch_engine.py
	torchEngineScript []byte
	//go:embed torch/wan22-vae-keys.json
	torchWanVAEKeys []byte
	//go:embed torch/umt5-xxl-keys.json
	torchUMT5Keys []byte
	//go:embed torch/flux2-vae-keys.json
	torchFluxVAEKeys []byte
)

// torchKeyTables are the key tables the engine script renames tensors by:
// the file's name in the engine folder, the name the load message gives it,
// the model set that uses it, and its bytes.
var torchKeyTables = []struct {
	name, table, set string
	data             []byte
}{
	{"wan22-vae-keys", "wan-vae", "hummingbird-xt-i2v", torchWanVAEKeys},
	{"umt5-xxl-keys", "umt5", "hummingbird-xt-i2v", torchUMT5Keys},
	{"flux2-vae-keys", "flux-vae", "flux2-klein-4b", torchFluxVAEKeys},
}

// A TorchSetPlace is one model set an OA PyTorch engine serves: the set, a
// person's options for it, where its files lie and the record of their
// placed proof (PlaceTorchModels).
type TorchSetPlace struct {
	Set                 TorchModelSet
	Options             map[string]string
	Places, ModelRecord string
}

// torchRuntime is the reviewed interpreter the engine runs on: the venv's
// own launcher by size and SHA-256, the base interpreter that launcher
// starts (pyvenv.cfg's home) by the files that run, the Python version
// pyvenv.cfg names, the venv's site-packages and ROCm library folders, and
// the GPU family the ROCm wheels were built for.
type torchRuntime struct {
	Release string
	Python  enginefiles.File
	// Base are the files of the base interpreter folder that run, by name
	// in that folder: the interpreter, its DLLs and the C runtime it
	// loads.
	Base          []enginefiles.File
	PythonVersion string
	// SitePackages is the venv's site-packages folder, a slash path below
	// the venv; the engine adds it to its own path (PYTORCH-T1).
	SitePackages string
	// Libraries is the folder of the ROCm libraries' bin below the venv, a
	// slash path, whose gfx kernel folders the path-length gate reads
	// (INF-EL3); empty for a stack with no ROCm libraries.
	Libraries    string
	TargetFamily string
}

// reviewedTorchRuntime is the stack of
// research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0.md:
// PyTorch 2.11.0 on ROCm 7.13.0 for gfx1151 in a venv of python.org's
// CPython 3.12.10, whose venv launcher is Scripts\python.exe. The base
// interpreter's files are python.org's 3.12.10 x64 build as installed on the
// owner's machine.
var reviewedTorchRuntime = torchRuntime{
	Release: "torch-2.11.0+rocm7.13.0 cpython-3.12.10 gfx1151",
	Python:  enginefiles.File{Path: "python.exe", Size: 274424, SHA256: "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"},
	Base: []enginefiles.File{
		{Path: "python.exe", Size: 104952, SHA256: "4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a"},
		{Path: "python312.dll", Size: 6945272, SHA256: "9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120"},
		{Path: "python3.dll", Size: 70376, SHA256: "fb975a606e7fbf74f64260e3f60c3490b4f74a183c0926fd6ed1ac4c52ac7b1c"},
		{Path: "vcruntime140.dll", Size: 120400, SHA256: "052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4"},
		{Path: "vcruntime140_1.dll", Size: 49776, SHA256: "6a99bc0128e0c7d6cbbf615fcc26909565e17d4ca3451b97f8987f9c6acbc6c8"},
		// The native modules the network audit hook's own import of socket
		// loads before the hook runs: _socket, and select through socket's
		// import of selectors. The other native modules of DLLs\ are the
		// accepted stdlib risk.
		{Path: "DLLs/_socket.pyd", Size: 85496, SHA256: "3efa4636cf9e350856f6927d67a4c2564bf9bd2758a00bbf17d3aece5b60016d"},
		{Path: "DLLs/select.pyd", Size: 33136, SHA256: "d4d7181b11db0bd22a6cd5cda6f8583d1f5c77615fd0a6ca33350619d6e0438a"},
	},
	PythonVersion: "3.12.10",
	SitePackages:  "Lib/site-packages",
	Libraries:     "Lib/site-packages/_rocm_sdk_libraries_gfx1151/bin",
	TargetFamily:  "gfx1151",
}

// pinnedTorchRuntime is the reviewed stack for a platform; only Windows x64
// has one.
func pinnedTorchRuntime(goos, goarch string) (torchRuntime, bool) {
	if goos == "windows" && goarch == "amd64" {
		return reviewedTorchRuntime, true
	}
	return torchRuntime{}, false
}

// A TorchEngine is the OA PyTorch engine serving one model set: the venv's
// interpreter, AMD's reviewed source folder, the model set and where its
// files lie (CONTRACT.md PYTORCH-T1..T9). Nothing here is downloaded or
// copied; the engine script is the provider's own.
type TorchEngine struct {
	// Python is the absolute path of the reviewed venv's interpreter:
	// <venv>\Scripts\python.exe on Windows, <venv>/bin/python elsewhere.
	Python string
	// Source is the absolute folder of AMD's Hummingbird-XT infer/ source
	// at the reviewed commit; the set's Source files are proved there.
	Source string
	// Set is the model set and Options the person's choices for it.
	Set     TorchModelSet
	Options map[string]string
	// Places and ModelRecord are where the set's files lie and the record of
	// their placed proof (PlaceTorchModels).
	Places, ModelRecord string
	// Beside, when set, is a second model set the same engine process
	// serves: either set loads first, and the other joins the running
	// process (PYTORCH-T10).
	Beside *TorchSetPlace
	// State is the folder in the provider's private state the engine script,
	// its tables, its record, its caches, its temporary folder and its
	// working folder live in.
	State string
	// Files proves and holds the engine's files and the model files.
	Files *enginefiles.Verifier
	// Ready bounds a load; zero reads DefaultVideoReady.
	Ready time.Duration
	// Log is the absolute file the engine's stderr is kept in.
	Log string
	// MemoryLimit, when not zero, is the job's per-process memory limit.
	MemoryLimit uint64
	// Sandbox is the manifest of the engine's sandbox Setup installed
	// (pytorchd setup install, PYTORCH-T12); every launch checks it and
	// starts the engine inside it. Windows refuses a launch without one.
	Sandbox string

	// runtime is the stack Python must be; zero reads the platform's pin.
	runtime torchRuntime
	// script replaces the embedded engine script; only tests set it.
	script []byte
	// fixture are the stand-in engine's own arguments; only tests set them.
	fixture []string
	// gpuProbe replaces the path-length gate's system answers; only tests
	// set it.
	gpuProbe   *enginelaunch.GPUProbe
	report     func(error)
	deviceLost func(p *Process, marker string)
	// exited receives the engine's exit record (KIT-P3).
	exited func(EngineExit)
}

// sets are the model sets the engine serves: its own, then Beside's.
func (e TorchEngine) sets() []TorchSetPlace {
	out := []TorchSetPlace{{Set: e.Set, Options: e.Options, Places: e.Places, ModelRecord: e.ModelRecord}}
	if e.Beside != nil {
		out = append(out, *e.Beside)
	}
	return out
}

// place is the served set named name.
func (e TorchEngine) place(name string) (TorchSetPlace, bool) {
	for _, sp := range e.sets() {
		if sp.Set.Name == name {
			return sp, true
		}
	}
	return TorchSetPlace{}, false
}

// needsSource reports whether a served set runs AMD's reviewed source.
func (e TorchEngine) needsSource() bool {
	for _, sp := range e.sets() {
		if len(sp.Set.Source) > 0 {
			return true
		}
	}
	return false
}

// configured refuses a configuration that names no reviewed stack or no
// placed set.
func (e TorchEngine) configured() error {
	if e.Python == "" {
		return errors.New("no PyTorch engine is configured; declare one with --python")
	}
	if !filepath.IsAbs(e.Python) || e.needsSource() && !filepath.IsAbs(e.Source) || !filepath.IsAbs(e.State) {
		return errors.New("the PyTorch interpreter, source folder and state folder must be absolute paths")
	}
	paths := []string{e.Python, e.State}
	if e.needsSource() {
		paths = append(paths, e.Source)
	}
	for _, p := range paths {
		if err := sandbox.CheckLocalPlacement(p); err != nil {
			return err
		}
	}
	if e.runtime.Release == "" {
		return fmt.Errorf("no reviewed PyTorch stack is pinned for %s/%s; this provider launches no other", runtime.GOOS, runtime.GOARCH)
	}
	if !identity.SamePathName(filepath.Base(e.Python), e.runtime.Python.Path) {
		return fmt.Errorf("the interpreter %s is not the venv's %s", e.Python, e.runtime.Python.Path)
	}
	if e.Beside != nil && e.Beside.Set.Name == e.Set.Name {
		return fmt.Errorf("the PyTorch engine names the model set %s twice", e.Set.Name)
	}
	for _, sp := range e.sets() {
		if err := sp.Set.valid(); err != nil {
			return err
		}
		if err := sp.Set.checkOptions(sp.Options); err != nil {
			return err
		}
		if !filepath.IsAbs(sp.Places) || !filepath.IsAbs(sp.ModelRecord) {
			return errors.New("the PyTorch model places and their record must be absolute paths")
		}
		for _, p := range []string{sp.Places, sp.ModelRecord} {
			if err := sandbox.CheckLocalPlacement(p); err != nil {
				return err
			}
		}
	}
	if e.Files == nil {
		return errors.New("the PyTorch engine has no verifier")
	}
	if _, err := torchFixes(); err != nil {
		return err
	}
	return nil
}

// venv is the folder the interpreter's venv is rooted at.
func (e TorchEngine) venv() string { return filepath.Dir(filepath.Dir(e.Python)) }

// torchFixes are the engine-side fixes of the resource kit's pytorch-rocm
// dispatch row, by the names the engine script applies them under. A row
// fix the engine does not know refuses every launch: a fix nobody applies is
// a known-bad dispatch left open (resource RES-G5).
func torchFixes() ([]string, error) {
	row, ok := gpufault.DispatchFor(gpufault.EnginePyTorchROCm)
	if !ok {
		return nil, errors.New("the resource kit has no pytorch-rocm dispatch row")
	}
	known := []struct{ prefix, name string }{
		{"torch.backends.cudnn.enabled=False", "cudnn-off"},
		{"Hummingbird-XT WanDiffusionWrapper.seq_len from the clip", "seq-len-from-clip"},
	}
	var out []string
	for _, fix := range row.Fixes {
		i := slices.IndexFunc(known, func(k struct{ prefix, name string }) bool { return strings.HasPrefix(fix, k.prefix) })
		if i < 0 {
			return nil, fmt.Errorf("the pytorch-rocm dispatch row asks for a fix the engine does not apply: %s", fix)
		}
		out = append(out, known[i].name)
	}
	sort.Strings(out)
	return out, nil
}

// torchOwners are the owners an engine file, a source file or the
// interpreter may have to be trusted in place: the machine's and this
// account.
func torchOwners() trustedpath.Owners {
	o := trustedpath.MachineOwners()
	o.Account = true
	return o
}

// engineFile is one file of the engine group: its placed proof and the role
// the load message names it by.
type engineFile struct {
	placed enginefiles.Placed
	role   string
	module string
	table  string
	// set is the model set a source file or key table belongs to.
	set string
}

// Roles of the engine group's files.
const (
	roleLauncher = "launcher"
	roleScript   = "script"
)

// engineFiles are the venv's launcher, its pyvenv.cfg, the base interpreter
// files that run, the engine script, its key tables and the set's source
// files, as the engine group proves them. pyvenv.cfg is read and checked
// first: it is the venv's own file, so its digest is what this read found,
// and the proof then holds those bytes. The base interpreter is the folder
// pyvenv.cfg names as home, which the launcher starts.
func (e TorchEngine) engineFiles() ([]engineFile, error) {
	cfg := filepath.Join(e.venv(), "pyvenv.cfg")
	data, err := os.ReadFile(cfg)
	if err != nil {
		return nil, fmt.Errorf("the interpreter %s is in no venv: %w", e.Python, err)
	}
	home, err := checkPyvenv(data, e.runtime.PythonVersion)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", cfg, err)
	}
	owners := torchOwners()
	cfgSum := sha256.Sum256(data)
	out := []engineFile{
		{role: roleLauncher, placed: enginefiles.Placed{Path: e.Python, Size: e.runtime.Python.Size, SHA256: e.runtime.Python.SHA256, Owners: owners}},
		{placed: enginefiles.Placed{Path: cfg, Size: int64(len(data)), SHA256: hex.EncodeToString(cfgSum[:]), Owners: owners}},
	}
	for _, f := range e.runtime.Base {
		out = append(out, engineFile{placed: enginefiles.Placed{Path: filepath.Join(home, filepath.FromSlash(f.Path)), Size: f.Size,
			SHA256: f.SHA256, Owners: owners}})
	}
	script := e.script
	if script == nil {
		script = torchEngineScript
	}
	p, sum, err := e.materialize("oa_torch_engine", script)
	if err != nil {
		return nil, err
	}
	out = append(out, engineFile{placed: enginefiles.Placed{Path: p, Size: int64(len(script)), SHA256: sum, Owners: owners}, role: roleScript})
	// The key tables and the source of every set the engine serves are held
	// for the process's life, so a set that joins a running engine reads
	// the files proved at its launch.
	for _, sp := range e.sets() {
		for _, t := range torchKeyTables {
			if t.set != sp.Set.Name {
				continue
			}
			p, sum, err := e.materialize(t.name, t.data)
			if err != nil {
				return nil, err
			}
			out = append(out, engineFile{placed: enginefiles.Placed{Path: p, Size: int64(len(t.data)), SHA256: sum, Owners: owners},
				table: t.table, set: sp.Set.Name})
		}
		for _, s := range sp.Set.Source {
			out = append(out, engineFile{placed: enginefiles.Placed{Path: filepath.Join(e.Source, filepath.FromSlash(s.Path)),
				Size: s.Size, SHA256: s.SHA256, Owners: owners}, module: s.Module, set: sp.Set.Name})
		}
	}
	return out, nil
}

// checkPyvenv refuses a pyvenv.cfg that names another Python version, lets
// the system's site-packages in, or names no absolute home, and returns the
// home.
func checkPyvenv(data []byte, version string) (string, error) {
	values := map[string]string{}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return "", fmt.Errorf("the line %q is not KEY = VALUE", line)
		}
		values[strings.ToLower(strings.TrimSpace(key))] = strings.TrimSpace(value)
	}
	switch {
	case values["version"] != version && values["version_info"] != version:
		return "", fmt.Errorf("the venv's Python is not %s", version)
	case !strings.EqualFold(values["include-system-site-packages"], "false"):
		return "", errors.New("the venv lets the system's site-packages in")
	case !filepath.IsAbs(values["home"]):
		return "", errors.New("the venv names no absolute home")
	}
	return filepath.Clean(values["home"]), nil
}

// sitePackages is the venv's site-packages folder, under the venv the held
// launcher's final path names.
func (e TorchEngine) sitePackages(launcher string) (string, error) {
	if e.runtime.SitePackages == "" {
		return "", errors.New("the reviewed stack names no site-packages folder")
	}
	site := filepath.Join(filepath.Dir(filepath.Dir(launcher)), filepath.FromSlash(e.runtime.SitePackages))
	if info, err := os.Stat(site); err != nil || !info.IsDir() {
		return "", fmt.Errorf("the venv has no site-packages folder at %s", site)
	}
	return site, nil
}

// errGPUPath is the refusal word of a launch the path-length gate refused
// (inference INF-EL3, PYTORCH-T4).
const errGPUPath = "engine:gpu-path"

// gpuGate runs the provider kit's path-length gate on the venv's hipBLASLt
// and rocBLAS kernel folders before every GPU launch (stack review condition
// 5) and returns the environment it needs: nothing, or each library's
// TENSILE_LIBPATH set to the verified 8.3 name of its folder.
func (e TorchEngine) gpuGate() ([]string, error) {
	if e.runtime.Libraries == "" {
		return nil, nil
	}
	bin := filepath.Join(e.venv(), filepath.FromSlash(e.runtime.Libraries))
	family := e.runtime.TargetFamily
	libraries := []enginelaunch.GPULibrary{
		{Dir: filepath.Join(bin, "hipblaslt", "library", family), EnvVar: "HIPBLASLT_TENSILE_LIBPATH", Extra: 1},
		{Dir: filepath.Join(bin, "rocblas", "library", family), EnvVar: "ROCBLAS_TENSILE_LIBPATH", Extra: 2},
	}
	probe := enginelaunch.SystemGPUProbe()
	if e.gpuProbe != nil {
		probe = *e.gpuProbe
	}
	env, err := enginelaunch.GPUPathGate(e.venv(), libraries, probe)
	if err != nil {
		return nil, refuse(errGPUPath, "the venv %s: %v; move it closer to the drive root", e.venv(), err)
	}
	return env, nil
}

// materialize writes data to the engine folder of State under a name bound
// to its digest, unless the file there already holds it, and returns its
// path and SHA-256.
func (e TorchEngine) materialize(name string, data []byte) (string, string, error) {
	sum := sha256.Sum256(data)
	digest := hex.EncodeToString(sum[:])
	ext := ".py"
	if name != "oa_torch_engine" {
		ext = ".json"
	}
	dir := filepath.Join(e.State, "engine")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", "", err
	}
	p := filepath.Join(dir, name+"-"+digest[:16]+ext)
	if held, err := os.ReadFile(p); err == nil && bytes.Equal(held, data) {
		return p, digest, nil
	}
	temp, err := os.CreateTemp(dir, name+"-*.part")
	if err != nil {
		return "", "", err
	}
	_, werr := temp.Write(data)
	cerr := temp.Close()
	if werr != nil || cerr != nil {
		//unchecked: the write failure is the error returned
		os.Remove(temp.Name())
		return "", "", errors.Join(werr, cerr)
	}
	if err := os.Rename(temp.Name(), p); err != nil {
		//unchecked: the rename failure is the error returned
		os.Remove(temp.Name())
		return "", "", err
	}
	return p, digest, nil
}

// engineRecord is the metadata record of the engine group's placed proof.
func (e TorchEngine) engineRecord() string { return filepath.Join(e.State, "engine.files") }

// modelPlaced is each pinned model file of sp where its places record says
// it lies, with its owners.
func modelPlaced(sp TorchSetPlace) ([]enginefiles.Placed, error) {
	where, owners, err := readFilePlaces(sp.Places, sp.Set.Name, sp.Set.Files)
	if err != nil {
		return nil, err
	}
	out := make([]enginefiles.Placed, 0, len(sp.Set.Files))
	for _, f := range sp.Set.Files {
		out = append(out, enginefiles.Placed{Path: where[f.Name], Size: f.Size, SHA256: f.SHA256, Owners: owners[f.Name]})
	}
	return out, nil
}

func placedOf(files []engineFile) []enginefiles.Placed {
	out := make([]enginefiles.Placed, len(files))
	for i, f := range files {
		out[i] = f.placed
	}
	return out
}

// present proves the interpreter, the engine's own files, the source and the
// placed model files of sp, and releases them at once; it is the only check
// before a lease is asked for.
func (e TorchEngine) present(ctx context.Context, sp TorchSetPlace) error {
	if err := e.configured(); err != nil {
		return err
	}
	files, err := e.engineFiles()
	if err != nil {
		return err
	}
	if err := e.Files.CheckPlaced(ctx, "the PyTorch engine", placedOf(files), e.engineRecord()); err != nil {
		return fmt.Errorf("the PyTorch interpreter, engine or source files are not the reviewed ones (%w)", err)
	}
	models, err := modelPlaced(sp)
	if err != nil {
		return err
	}
	if err := e.Files.CheckPlaced(ctx, sp.Set.Name, models, sp.ModelRecord); err != nil {
		return fmt.Errorf("the %s files are not the pinned ones where they were placed (%w); place them again with pytorchd models install --set %s", sp.Set.Name, err, sp.Set.Name)
	}
	if _, err := e.gpuGate(); err != nil {
		return err
	}
	return e.checkSandbox(files)
}

// checkSandbox checks the engine's sandbox for the engine files as placed,
// before any lease (PYTORCH-T12); a launch checks it again on the files it
// holds.
func (e TorchEngine) checkSandbox(files []engineFile) error {
	site, err := e.sitePackages(e.Python)
	if err != nil {
		return refuse(NoEngine, "%v", err)
	}
	paths := make([]string, len(files))
	for i, f := range files {
		paths[i] = f.placed.Path
	}
	launch, err := e.prepareSandbox(paths, site)
	launch.Close()
	return err
}

// CheckTorchEngine proves the interpreter at python, the engine's own files
// in state and set's source in source as a video does before its lease,
// checks the engine's sandbox the manifest sandboxManifest records, and
// releases them.
func CheckTorchEngine(ctx context.Context, python, source, state, sandboxManifest string, set TorchModelSet) error {
	//unchecked: a platform with no pin leaves the runtime zero, which is refused by name below
	rt, _ := pinnedTorchRuntime(runtime.GOOS, runtime.GOARCH)
	e := TorchEngine{Python: python, Source: source, State: state, Set: set, runtime: rt, Sandbox: sandboxManifest,
		Files: enginefiles.NewVerifier(enginefiles.Options{})}
	switch {
	case !filepath.IsAbs(python) || len(set.Source) > 0 && !filepath.IsAbs(source) || !filepath.IsAbs(state):
		return errors.New("the interpreter, the source folder and the state folder must be absolute paths")
	case rt.Release == "":
		return fmt.Errorf("no reviewed PyTorch stack is pinned for %s/%s", runtime.GOOS, runtime.GOARCH)
	case !identity.SamePathName(filepath.Base(python), rt.Python.Path):
		return fmt.Errorf("the interpreter %s is not the venv's %s", python, rt.Python.Path)
	}
	if err := set.valid(); err != nil {
		return err
	}
	files, err := e.engineFiles()
	if err != nil {
		return err
	}
	if err := e.Files.CheckPlaced(ctx, "the PyTorch engine", placedOf(files), e.engineRecord()); err != nil {
		return err
	}
	if _, err := e.gpuGate(); err != nil {
		return err
	}
	return e.checkSandbox(files)
}

// CheckTorchModels proves a PyTorch set's files where places says they lie,
// as a video does before its lease.
func CheckTorchModels(ctx context.Context, files *enginefiles.Verifier, set TorchModelSet, places, record string) error {
	placed, err := modelPlaced(TorchSetPlace{Set: set, Places: places})
	if err != nil {
		return err
	}
	return files.CheckPlaced(ctx, set.Name, placed, record)
}

// torchSystemNames are the Windows variables the engine receives beside the
// clean environment's own, each computed by the provider kit from the
// kernel, the token or the registry: the research launcher's allowlist
// (stack review condition 3).
var torchSystemNames = []string{"USERPROFILE", "LOCALAPPDATA", "APPDATA", "ProgramData", "ALLUSERSPROFILE",
	"NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "OS", "USERNAME", "ComSpec", "PATHEXT"}

// torchEnvironmentNames are every name the engine environment may hold.
var torchEnvironmentNames = map[string]bool{
	"SYSTEMROOT": true, "WINDIR": true, "SYSTEMDRIVE": true, "PATH": true, "TEMP": true, "TMP": true, "TMPDIR": true,
	"USERPROFILE": true, "LOCALAPPDATA": true, "APPDATA": true, "PROGRAMDATA": true, "ALLUSERSPROFILE": true,
	"NUMBER_OF_PROCESSORS": true, "PROCESSOR_ARCHITECTURE": true, "PROCESSOR_IDENTIFIER": true, "OS": true,
	"USERNAME": true, "COMSPEC": true, "PATHEXT": true,
	"ROCM_SDK_TARGET_FAMILY": true,
	"HF_HUB_OFFLINE":         true, "TRANSFORMERS_OFFLINE": true, "HF_DATASETS_OFFLINE": true, "HF_HUB_DISABLE_TELEMETRY": true,
	"HF_HUB_DISABLE_IMPLICIT_TOKEN": true, "DO_NOT_TRACK": true, "HF_HOME": true, "TORCH_HOME": true,
	"TORCHINDUCTOR_CACHE_DIR": true, "TRITON_CACHE_DIR": true, "MIOPEN_USER_DB_PATH": true, "MIOPEN_CUSTOM_CACHE_DIR": true,
	"AMD_COMGR_CACHE_DIR": true, "TORCH_COMPILE_DISABLE": true, "TORCHDYNAMO_DISABLE": true,
	// The path-length gate's verified 8.3 names of the kernel folders.
	"HIPBLASLT_TENSILE_LIBPATH": true, "ROCBLAS_TENSILE_LIBPATH": true,
}

// torchForbidden refuses a name whatever the allowlist says: one that loads
// code or libraries, redirects Python's own search, or sends traffic through
// a proxy (stack review condition 3).
func torchForbidden(upper string) bool {
	switch {
	case strings.HasPrefix(upper, "PYTHON"), upper == "ROCM_SDK_PRELOAD_LIBRARIES", upper == "HIP_VISIBLE_DEVICES",
		strings.HasSuffix(upper, "_PROXY"), upper == "LD_PRELOAD", upper == "LD_LIBRARY_PATH", upper == "HF_TOKEN":
		return true
	}
	return false
}

// environment is the engine's environment: the provider kit's clean
// environment with the research launcher's allowlist, PATH of the venv's
// scripts and the system folders, the offline hub variables, caches under
// State, the pytorch-rocm dispatch row's variables (PYTORCH-T4), and the
// path-length gate's 8.3 library paths in gate.
func (e TorchEngine) environment(gate []string) ([]string, []error, error) {
	temp := filepath.Join(e.State, "temp")
	cache := filepath.Join(e.State, "cache")
	for _, d := range []string{temp, cache} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			return nil, nil, err
		}
	}
	set := []string{
		"HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
		"HF_HUB_DISABLE_IMPLICIT_TOKEN=1", "DO_NOT_TRACK=1",
		"HF_HOME=" + filepath.Join(cache, "hf"), "TORCH_HOME=" + filepath.Join(cache, "torch"),
		"TORCHINDUCTOR_CACHE_DIR=" + filepath.Join(cache, "inductor"), "TRITON_CACHE_DIR=" + filepath.Join(cache, "triton"),
		"MIOPEN_USER_DB_PATH=" + filepath.Join(cache, "miopen"), "MIOPEN_CUSTOM_CACHE_DIR=" + filepath.Join(cache, "miopen"),
		"AMD_COMGR_CACHE_DIR=" + filepath.Join(cache, "comgr"),
		"TORCH_COMPILE_DISABLE=1", "TORCHDYNAMO_DISABLE=1",
		// rocm_sdk resolves the GPU family from this instead of running
		// offload-arch from PATH.
		"ROCM_SDK_TARGET_FAMILY=" + e.runtime.TargetFamily,
	}
	if row, ok := gpufault.DispatchFor(gpufault.EnginePyTorchROCm); ok {
		set = append(set, row.Environment()...)
	}
	set = append(set, gate...)
	var omitted []error
	env, err := enginelaunch.CleanEnvironment{Temp: temp, Names: torchSystemNames, Registry: registryString, Set: set,
		OnError: func(err error) { omitted = append(omitted, err) }}.Build()
	if err != nil {
		return nil, nil, err
	}
	scripts := filepath.Dir(e.Python)
	for i, item := range env {
		name, value, _ := strings.Cut(item, "=")
		if strings.EqualFold(name, "PATH") {
			if runtime.GOOS == "windows" {
				windows := filepath.Dir(value)
				env[i] = name + "=" + scripts + ";" + value + ";" + windows
			} else {
				env[i] = name + "=" + scripts + ":" + value
			}
		}
	}
	if err := checkTorchEnvironment(env); err != nil {
		return nil, nil, err
	}
	return env, omitted, nil
}

// checkTorchEnvironment refuses a name outside the allowlist, a forbidden
// name, an empty value and a repeated name.
func checkTorchEnvironment(env []string) error {
	seen := map[string]bool{}
	row, _ := gpufault.DispatchFor(gpufault.EnginePyTorchROCm)
	for _, item := range env {
		name, value, ok := strings.Cut(item, "=")
		upper := strings.ToUpper(name)
		_, fromRow := row.Env[name]
		if !ok || !(torchEnvironmentNames[upper] || fromRow) || torchForbidden(upper) || seen[upper] || value == "" {
			return fmt.Errorf("%w: %q", ErrEngineEnvironment, name)
		}
		seen[upper] = true
	}
	if len(env) == 0 {
		return fmt.Errorf("%w: empty", ErrEngineEnvironment)
	}
	return nil
}

// / torchProcess is one running engine, the provider's end of its pipes, and
// what the process holds: the engine group's files for its life, and each
// loaded set's model files until that set unloads.
type torchProcess struct {
	process *Process
	conn    *torchConn
	engine  TorchEngine
	files   []engineFile
	finals  []string
	fixes   []string
	site    string
	// contained is an engine running in its sandbox, which is lent its
	// model files rather than given their paths (PYTORCH-T12).
	contained bool
	// bootstrap is the post-spawn hello and first load. Holder owns the
	// process and lease before Ready calls it, including on its failure.
	bootstrap func(context.Context) error

	mu sync.Mutex
	// holds are each set's held model files, by set name.
	holds map[string]*enginefiles.Hold
	ended bool
}

// exited reports whether the process is gone.
func (t *torchProcess) exited() bool {
	select {
	case <-t.process.exited:
		return true
	default:
		return false
	}
}

// keep records set's model hold; a process already gone releases it.
func (t *torchProcess) keep(set string, hold *enginefiles.Hold) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.ended {
		return false
	}
	t.holds[set] = hold
	return true
}

// release lets go of set's model files.
func (t *torchProcess) release(set string) {
	t.mu.Lock()
	hold := t.holds[set]
	delete(t.holds, set)
	t.mu.Unlock()
	if hold != nil {
		//unchecked: the set is unloaded; a failed close leaves only read-shared handles of this process
		hold.Release()
	}
}

// releaseAll lets go of every set's model files once the process is gone.
func (t *torchProcess) releaseAll() error {
	t.mu.Lock()
	t.ended = true
	holds := t.holds
	t.holds = map[string]*enginefiles.Hold{}
	t.mu.Unlock()
	var failures []error
	for _, hold := range holds {
		if err := hold.Release(); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

// start proves and holds the engine's files and sp's model files, starts
// the interpreter on the held script with the clean environment and the
// protocol pipes, and sends sp's load (PYTORCH-T1..T5). Everything held is
// released once the engine's exit is confirmed.
func (e TorchEngine) start(sp TorchSetPlace) (*torchProcess, error) {
	if err := e.configured(); err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	fixes, err := torchFixes()
	if err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	files, err := e.engineFiles()
	if err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	models, err := modelPlaced(sp)
	if err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	// The path-length gate runs before every launch, as the GPU libraries
	// open their kernel files on the GPU's first use (stack review
	// condition 5, inference INF-EL3).
	gate, err := e.gpuGate()
	if err != nil {
		return nil, err
	}
	env, omitted, err := e.environment(gate)
	if err != nil {
		return nil, refuse(Unavailable, "the engine environment: %v", err)
	}
	if e.report != nil {
		for _, diagnostic := range omitted {
			e.report(diagnostic)
		}
	}
	work := filepath.Join(e.State, "work")
	if err := sandbox.CheckLocalPlacement(work); err != nil {
		return nil, refuse(Unavailable, "the engine's working folder: %v", err)
	}
	engineHold, err := e.Files.LaunchPlaced(context.Background(), "the PyTorch engine", placedOf(files), e.engineRecord())
	if err != nil {
		return nil, refuse(NoEngine, "the PyTorch interpreter, engine or source files: %v", err)
	}
	modelHold, err := e.Files.LaunchPlaced(context.Background(), sp.Set.Name, models, sp.ModelRecord)
	if err != nil {
		//unchecked: the refusal below is the error returned; a failed close leaves only read-shared handles
		engineHold.Release()
		return nil, refuse(NoEngine, "the %s files: %v", sp.Set.Name, err)
	}
	release := func() error {
		return errors.Join(engineHold.Release(), modelHold.Release())
	}
	// The engine opens every file by the final path of the handle held for
	// it, every folder on which is held and no reparse point.
	finals := engineHold.Paths()
	launcher, script := finalOf(files, finals, roleLauncher), finalOf(files, finals, roleScript)
	site, err := e.sitePackages(launcher)
	if err != nil {
		release()
		return nil, refuse(NoEngine, "%v", err)
	}
	if launcher == "" || script == "" || len(finals) != len(files) {
		release()
		return nil, refuse(Unavailable, "the held engine files do not match the proof")
	}
	// The sandbox is checked on the files now held, and held shared until
	// the engine has ended, so Setup cannot change it under the engine
	// (PYTORCH-T12, KIT-S3).
	launch, err := e.prepareSandbox(finals, site)
	if err != nil {
		release()
		return nil, err
	}
	release = func() error {
		err := errors.Join(engineHold.Release(), modelHold.Release())
		launch.Close()
		return err
	}
	t := &torchProcess{engine: e, files: files, finals: finals, fixes: fixes, site: site, holds: map[string]*enginefiles.Hold{},
		contained: launch != nil}
	var container string
	if launch != nil {
		container = launch.SID
	}
	var log *engineLog
	var output io.Writer
	if e.Log != "" {
		if log, err = openEngineLog(e.Log, engineLogBytes); err != nil {
			release()
			return nil, refuse(Unavailable, "the engine log: %v", err)
		}
		output = log
	}
	closeLog := func() error {
		if log != nil {
			return log.Close()
		}
		return nil
	}
	lostOutput, lost := engineOutput(output, devicelost.PyTorchEngine, e.deviceLost)
	if lostOutput == nil {
		lostOutput = io.Discard
	}
	inRead, inWrite, err := os.Pipe()
	if err != nil {
		release()
		closeLog()
		return nil, refuse(Unavailable, "the engine's pipes: %v", err)
	}
	outRead, outWrite, err := os.Pipe()
	if err != nil {
		closePipes(inRead, inWrite)
		release()
		closeLog()
		return nil, refuse(Unavailable, "the engine's pipes: %v", err)
	}
	// -I: isolated (no PYTHON* variable, no user site, no script folder on
	// the path); -S: no site module, so no .pth file runs and no
	// site-packages is on the path until the engine adds the venv's after
	// its network hook (PYTORCH-T1); -B: no bytecode written.
	args := append([]string{"-I", "-S", "-B", "-X", "utf8", script}, e.fixture...)
	// The exit record names the set the process was started for, which is
	// the image set when an image starts the process its video set joins
	// (KIT-P3, PYTORCH-T10).
	p, err := startOwned(ownedSpec{Program: launcher, Args: args, Env: env, Dir: work, Stdin: inRead, Stdout: outWrite,
		Output: lostOutput, MemoryLimit: e.MemoryLimit, Engine: "oa-torch-engine", Model: sp.Set.Name, Exited: e.exited,
		Container: container})
	closePipes(inRead, outWrite)
	if err != nil {
		closePipes(inWrite, outRead)
		release()
		closeLog()
		if container != "" {
			return nil, refuse(EngineSandbox, "the engine's start in its sandbox: %v", err)
		}
		return nil, err
	}
	conn := newTorchConn(outRead, inWrite, sp.Set.Name)
	t.process, t.conn = p, conn
	// A holder observes p.exited. Publish that only after the file holds,
	// sandbox lock and log have been released, so Close followed immediately
	// by Setup/remove cannot race this cleanup goroutine.
	processExited := p.exited
	cleaned := make(chan struct{})
	p.exited = cleaned
	go func() {
		<-processExited
		closePipes(inWrite, outRead)
		cleanupErr := errors.Join(release(), t.releaseAll(), closeLog())
		if cleanupErr != nil && e.report != nil {
			e.report(fmt.Errorf("pytorch: post-exit cleanup of the engine: %w", cleanupErr))
		}
		close(cleaned)
	}()
	t.bootstrap = func(ctx context.Context) error {
		// Python finishes configuring its standard handles before hello. A
		// lent handle during startup can be reused by Python as a pipe.
		timer := time.NewTimer(10 * time.Second)
		defer timer.Stop()
		select {
		case <-conn.hello:
			// The PID is a selector; LendTo proves its live job and SID.
		case <-conn.closed:
			return refuse(Unavailable, "the engine ended before its hello: %v", conn.broken())
		case <-timer.C:
			return refuse(Unavailable, "the engine did not send its hello")
		case <-ctx.Done():
			return ctx.Err()
		}
		load, err := t.loadMessage(sp, modelHold.Paths())
		if err != nil {
			return refuse(Unavailable, "the load message: %v", err)
		}
		// An engine whose protocol closes after loading is stopped by its
		// owner, which retains the lease until its exit is confirmed.
		go func() {
			select {
			case <-conn.closed:
			case <-p.exited:
				return
			}
			if err := conn.broken(); err != nil && !errors.Is(err, io.EOF) && e.report != nil {
				e.report(fmt.Errorf("pytorch: the PyTorch engine %d: %v", p.PID(), err))
			}
			p.stop(stopWithin)
		}()
		go func() {
			if err := conn.send(load, nil); err != nil {
				p.stop(stopWithin)
			}
		}()
		p.probe = func(ctx context.Context, _ *http.Client) bool {
			return conn.readyNow() && conn.ping(ctx)
		}
		lost.Bind(p)
		return nil
	}
	return t, nil
}

// add proves and holds sp's model files and asks the running engine to load
// sp beside the sets it holds (PYTORCH-T10). The files stay held until sp
// unloads or the engine exits.
func (t *torchProcess) add(sp TorchSetPlace) (*torchSetState, error) {
	if t.exited() {
		return nil, errTorchEngineChanged
	}
	models, err := modelPlaced(sp)
	if err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	hold, err := t.engine.Files.LaunchPlaced(context.Background(), sp.Set.Name, models, sp.ModelRecord)
	if err != nil {
		return nil, refuse(NoEngine, "the %s files: %v", sp.Set.Name, err)
	}
	if !t.keep(sp.Set.Name, hold) {
		//unchecked: the engine is gone; a failed close leaves only read-shared handles of this process
		hold.Release()
		return nil, errTorchEngineChanged
	}
	load, err := t.loadMessage(sp, hold.Paths())
	if err != nil {
		t.release(sp.Set.Name)
		return nil, refuse(Unavailable, "the load message: %v", err)
	}
	state := t.conn.expect(sp.Set.Name)
	if err := t.conn.send(load, nil); err != nil {
		t.release(sp.Set.Name)
		return nil, errTorchEngineChanged
	}
	return state, nil
}

// errTorchEngineChanged is a load planned for an engine that exited or was
// replaced before it ran; the load is planned again.
var errTorchEngineChanged = errors.New("the PyTorch engine changed while a model set was being added")

func closePipes(files ...*os.File) {
	for _, f := range files {
		//unchecked: a pipe end the engine no longer uses; a failed close loses nothing
		f.Close()
	}
}

// finalOf is the held final path of the engine file of role, or "".
func finalOf(files []engineFile, finals []string, role string) string {
	for i, f := range files {
		if f.role == role && i < len(finals) {
			return finals[i]
		}
	}
	return ""
}

// loadMessage is the engine's load of sp: the set, the venv's site-packages
// the engine adds to its path, the held final path of each file by role,
// the set's source modules and key tables, the options, the dispatch row's
// fixes and the preset.
func (t *torchProcess) loadMessage(sp TorchSetPlace, modelFinals []string) (map[string]any, error) {
	if len(t.finals) != len(t.files) || len(modelFinals) != len(sp.Set.Files) {
		return nil, errors.New("the held files do not match the proof")
	}
	source := map[string]any{}
	tables := map[string]any{}
	for i, f := range t.files {
		if f.set != sp.Set.Name {
			continue
		}
		switch {
		case f.module != "":
			source[f.module] = map[string]any{"path": t.finals[i], "sha256": f.placed.SHA256}
		case f.table != "":
			tables[f.table] = map[string]any{"path": t.finals[i], "sha256": f.placed.SHA256}
		}
	}
	options := map[string]any{}
	for _, o := range sp.Set.Options {
		options[strings.ReplaceAll(o.Name, "-", "_")] = sp.Set.option(sp.Options, o.Name)
	}
	p := sp.Set.Preset
	preset := map[string]any{"guidance": p.Guidance}
	if sp.Set.Profile == torchProfileImage {
		preset["steps"] = p.Steps
	} else {
		preset["timesteps"], preset["shift"], preset["boundary"] = p.Timesteps, p.Shift, p.Boundary
	}
	// The model files come last: in the sandbox each is a handle lent to
	// the engine, which nothing after it may fail to send.
	modelFiles, err := t.modelFiles(sp, modelFinals)
	if err != nil {
		return nil, err
	}
	return map[string]any{"op": "load", "set": sp.Set.Name, "site_packages": t.site, "files": modelFiles, "source": source,
		"tables": tables, "options": options, "fixes": t.fixes, "preset": preset}, nil
}
