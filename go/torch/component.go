package torch

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
)

// leanLock is the stack review's lean lock
// (research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0-lean.lock):
// the 25 distributions Setup installs, each by its wheel's SHA-256.
//
//go:embed torch/lean.lock
var leanLock []byte

// A Component is the PyTorch engine Setup installs for this account (TODO
// D5a, PYTORCH-T12). Its root holds the sandbox's manifest and lock beside
// engine\, the one folder the engine's sandbox reads: python\, a copy of
// python.org's CPython 3.12.10 for Windows x64 without site-packages, which
// the reviewed interpreter's pins prove; venv\, made from it without pip
// and holding the lean lock's distributions; and source\ when Setup places
// AMD's Hummingbird-XT infer/ there.
type Component struct {
	Root string
	// ExpectedRootKey is Setup's durable identity of a generated root. Only
	// removal uses it; an empty key keeps the manifest-required manual path.
	ExpectedRootKey string
}

// DefaultComponentRoot is %LOCALAPPDATA%\OpenAbstractions\engines\pytorch,
// per user, no administrator.
func DefaultComponentRoot() (string, error) {
	local, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("no local application data folder; pass --root: %w", err)
	}
	return filepath.Join(local, "OpenAbstractions", "engines", "pytorch"), nil
}

func (c Component) Manifest() string  { return filepath.Join(c.Root, "sandbox.json") }
func (c Component) EngineDir() string { return filepath.Join(c.Root, "engine") }
func (c Component) BaseDir() string   { return filepath.Join(c.EngineDir(), "python") }
func (c Component) Venv() string      { return filepath.Join(c.EngineDir(), "venv") }

// Python is the venv's interpreter, the --python a component's engine runs.
func (c Component) Python() string {
	if runtime.GOOS == "windows" {
		return filepath.Join(c.Venv(), "Scripts", "python.exe")
	}
	return filepath.Join(c.Venv(), "bin", "python")
}

// Profile is the component's sandbox profile: oa.engine.pytorch.torch for
// the default root, and the same with a label of its root for another,
// so an isolated install never shares or deletes the person's (KIT-S1).
// An installed component must have a manifest naming this root's profile.
func (c Component) Profile() (string, error) {
	var expected string
	// An existing root is keyed by the filesystem's own identity. Unicode
	// case folding (and even PathNameKey in a case-sensitive directory) can
	// join two distinct roots into one AppContainer profile.
	key, err := identity.FileKey(c.Root)
	if errors.Is(err, os.ErrNotExist) {
		key = filepath.Clean(c.Root)
	} else if err != nil {
		return "", err
	}
	if def, err := DefaultComponentRoot(); err == nil {
		if filepath.Clean(c.Root) == filepath.Clean(def) {
			expected, err = SandboxProfile("")
			if err != nil {
				return "", err
			}
		} else if same, err := identity.SameFile(c.Root, def); err == nil && same {
			expected, err = SandboxProfile("")
			if err != nil {
				return "", err
			}
		}
	}
	if expected == "" {
		sum := sha256.Sum256([]byte(key))
		var err error
		// The 40-hex label fills CreateAppContainerProfile's 64-character
		// limit and leaves a 160-bit root-identity digest.
		expected, err = SandboxProfile(hex.EncodeToString(sum[:])[:40])
		if err != nil {
			return "", err
		}
	}
	m, err := sandbox.ReadManifest(c.Manifest())
	if err == nil {
		if m.Profile != expected {
			return "", fmt.Errorf("the sandbox at %s belongs to %s, not this component's profile %s", c.Manifest(), m.Profile, expected)
		}
	} else if !errors.Is(err, sandbox.ErrNotInstalled) {
		return "", err
	}
	return expected, nil
}

// ComponentInstall is one install or update of the component.
type ComponentInstall struct {
	Component
	// PythonHome is the folder of python.org's CPython 3.12.10 for Windows
	// x64 whose files the reviewed interpreter's pins name.
	PythonHome string
	// Wheels are the folders holding the lean lock's wheels; nothing is
	// downloaded.
	Wheels []string
	// State is the engine's state folder, the provider's state and torch
	// (TorchEngine.State), whose script, cache, temporary and working
	// folders the sandbox is granted.
	State string
	// Source is AMD's Hummingbird-XT infer/ folder when it lies outside the
	// engine folder, granted read and execute; empty for none.
	Source string
	// Compile byte-compiles the venv's site-packages, as pip does.
	Compile bool
	// Out receives one line per step.
	Out io.Writer

	// runtime is the stack the copy and the venv must be; zero reads the
	// platform's pin. lock is the lock installed; nil reads the lean lock.
	// Only tests set them.
	runtime torchRuntime
	lock    []byte
}

// lockMarker records in the venv which lock it holds, so an update with
// the same lock installs nothing again.
const lockMarker = "oa-lock.sha256"

// InstallComponent installs or updates the PyTorch engine (TODO D5a): under
// the sandbox's lock, which refuses while an engine runs and refuses every
// launch until it finishes, it copies the base interpreter and proves it by
// its pins, makes the venv without pip and proves its launcher, installs
// the lock's wheels from local folders with every file checked against its
// RECORD, makes the engine's state folders, and applies the sandbox's
// grants (PYTORCH-T12, KIT-S2). Nothing is downloaded and nothing in a
// wheel runs.
func InstallComponent(ctx context.Context, o ComponentInstall) error {
	if o.runtime.Release == "" {
		o.runtime, _ = pinnedTorchRuntime(runtime.GOOS, runtime.GOARCH)
		if o.runtime.Release == "" {
			return fmt.Errorf("no reviewed PyTorch stack is pinned for %s/%s", runtime.GOOS, runtime.GOARCH)
		}
	}
	if o.lock == nil {
		o.lock = leanLock
	}
	if o.Out == nil {
		o.Out = io.Discard
	}
	paths := append([]string{o.Root, o.PythonHome, o.State, o.EngineDir(), o.BaseDir(), o.BaseDir() + ".part",
		o.Venv(), o.Python(), o.sitePackages(), o.Manifest()}, o.Wheels...)
	paths = append(paths, sandboxFolders(o.State)...)
	if o.Source != "" {
		paths = append(paths, o.Source)
	}
	for _, p := range paths {
		if err := sandbox.CheckLocalPlacement(p); err != nil {
			return err
		}
	}
	if len(o.Wheels) == 0 {
		return errors.New("name the folders that hold the lock's wheels")
	}
	entries, err := parseLock(o.lock)
	if err != nil {
		return err
	}
	// The root itself is the first effect. When it was absent, no persisted
	// manifest could own it; after creation Profile keys its actual file ID.
	if err := os.MkdirAll(o.Root, 0o700); err != nil {
		return err
	}
	for _, p := range paths {
		if err := sandbox.CheckLocalPlacement(p); err != nil {
			return err
		}
	}
	profile, err := o.Profile()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(o.EngineDir(), 0o700); err != nil {
		return err
	}
	if err := sandbox.CheckLocalPlacement(o.EngineDir()); err != nil {
		return err
	}
	s, err := sandbox.Begin(o.Manifest(), profile)
	if errors.Is(err, sandbox.ErrInUse) {
		return fmt.Errorf("%w (pytorchd serve runs the engine; stop it, or wait for its idle time, then run Setup again)", err)
	}
	if err != nil {
		return err
	}
	defer s.Close()
	say := func(format string, a ...any) {
		//unchecked: progress lines; the returned error reports the outcome
		fmt.Fprintf(o.Out, format+"\n", a...)
	}
	say("the engine's sandbox: profile %s, phase installing", profile)
	if err := o.installInterpreter(); err != nil {
		return err
	}
	say("the base interpreter %s is the reviewed one", o.BaseDir())
	if err := o.installVenv(ctx); err != nil {
		return err
	}
	say("the venv %s is the reviewed one", o.Venv())
	installed, err := o.installLock(ctx, entries)
	if err != nil {
		return err
	}
	if installed {
		say("installed %d distributions of the lock into %s, every file checked against its RECORD", len(entries), o.sitePackages())
	} else {
		say("the venv already holds this lock")
	}
	for _, d := range sandboxFolders(o.State) {
		if err := os.MkdirAll(d, 0o700); err != nil {
			return err
		}
	}
	source := o.Source
	if source != "" {
		if inside, err := identity.Within(source, o.EngineDir()); err == nil && inside {
			source = ""
		}
	}
	if err := s.Grant(SandboxGrants(o.EngineDir(), o.State, source)...); err != nil {
		return err
	}
	if err := s.Commit(); err != nil {
		return err
	}
	say("the engine's sandbox is ready: %s", o.Manifest())
	return nil
}

func (o ComponentInstall) sitePackages() string {
	return filepath.Join(o.Venv(), filepath.FromSlash(o.runtime.SitePackages))
}

// baseExcluded are the folders of a python.org install the engine never
// uses: the test suite, IDLE, Tk, the demos, pip's bundled wheels, and
// whatever the person installed into its site-packages.
var baseExcluded = []string{"Lib/test", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "Lib/ensurepip", "Lib/site-packages"}

// keepBase is the interpreter copy policy shared by installation and input
// packaging. Package preparation never includes the person's site-packages.
func keepBase(rel string, directory bool) bool {
	if directory {
		return (strings.Contains(rel, "/") || rel == "DLLs" || rel == "Lib") && !slices.Contains(baseExcluded, rel)
	}
	return strings.Contains(rel, "/") || strings.HasSuffix(rel, ".exe") || strings.HasSuffix(rel, ".dll") || rel == "LICENSE.txt"
}

// ComponentInputs is the read-only roster of the reviewed local inputs used
// to prepare an installed component package. Source is the Hummingbird-XT
// repository at the review's commit; its infer/ and licence files are kept.
// It returns the existing enginefiles manifest and creates no component,
// interpreter, venv, profile, grants or process.
func ComponentInputs(ctx context.Context, pythonHome string, wheels []string, source string) (enginefiles.Manifest, error) {
	rt, ok := pinnedTorchRuntime(runtime.GOOS, runtime.GOARCH)
	if !ok {
		return enginefiles.Manifest{}, fmt.Errorf("no reviewed PyTorch stack is pinned for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	// These are LICENSE and NOTICES of the same reviewed Hummingbird commit
	// whose eight infer modules HummingbirdXTI2VTorch.Source already pins.
	notices := []enginefiles.File{
		{Path: "LICENSE", Size: 11328, SHA256: "f827333cb365d1beac339a38099c416c0d74590a9c279172929e92e683a5c7a6"},
		{Path: "NOTICES", Size: 30932, SHA256: "558edf09969b2c9162955699a3dd5836cc423bc62fde30d5a458abfef63cf587"},
	}
	return componentInputs(ctx, pythonHome, wheels, source, rt, leanLock, HummingbirdXTI2VTorch.Source, notices)
}

// inputFilePin measures the bytes which supply the manifest entry in one
// read. A fixed owner pin is checked against those same bytes; wheel locks
// bind the full bytes by SHA-256 and have no separate size pin (Size -1).
func inputFilePin(file, name string, expected *enginefiles.File) (enginefiles.File, error) {
	var pin enginefiles.File
	entry, err := os.Lstat(file)
	if err != nil || !entry.Mode().IsRegular() {
		return pin, fmt.Errorf("input %s is not a regular file", name)
	}
	stream, err := os.Open(file)
	if err != nil {
		return pin, err
	}
	defer stream.Close()
	info, err := stream.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return pin, fmt.Errorf("input %s is not a regular file", name)
	}
	hash := sha256.New()
	size, err := io.Copy(hash, stream)
	if err != nil {
		return pin, err
	}
	pin = enginefiles.File{Path: name, Size: size, SHA256: hex.EncodeToString(hash.Sum(nil))}
	if expected != nil && (pin.SHA256 != expected.SHA256 || expected.Size >= 0 && pin.Size != expected.Size) {
		return enginefiles.File{}, fmt.Errorf("input %s changed from its reviewed pin", name)
	}
	return pin, nil
}

func componentInputs(ctx context.Context, pythonHome string, folders []string, source string, rt torchRuntime, lock []byte, sources []TorchSourceFile, notices []enginefiles.File) (enginefiles.Manifest, error) {
	var manifest enginefiles.Manifest
	if err := ctx.Err(); err != nil {
		return manifest, err
	}
	paths := append([]string{pythonHome, source}, folders...)
	if len(folders) == 0 {
		return manifest, errors.New("name the folders that hold the lock's wheels")
	}
	for _, p := range paths {
		if !filepath.IsAbs(p) {
			return manifest, errors.New("input folders must be absolute local paths")
		}
		if err := sandbox.CheckLocalPlacement(p); err != nil {
			return manifest, err
		}
	}
	if err := (ComponentInstall{runtime: rt}).verifyBase(pythonHome); err != nil {
		return manifest, fmt.Errorf("the input interpreter differs from the reviewed CPython: %w", err)
	}
	entries, err := parseLock(lock)
	if err != nil {
		return manifest, err
	}
	selected, err := lockedWheels(entries, folders)
	if err != nil {
		return manifest, err
	}
	add := func(file, name string, expected *enginefiles.File) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		pin, err := inputFilePin(file, name, expected)
		if err != nil {
			return err
		}
		manifest.Files = append(manifest.Files, pin)
		return nil
	}
	var base func(string, string) error
	base = func(folder, rel string) error {
		list, err := os.ReadDir(folder)
		if err != nil {
			return err
		}
		for _, entry := range list {
			name := entry.Name()
			if rel != "" {
				name = rel + "/" + name
			}
			if entry.Type()&(fs.ModeSymlink|fs.ModeIrregular) != 0 {
				return fmt.Errorf("%s is a link or a junction", filepath.Join(folder, entry.Name()))
			}
			if !keepBase(name, entry.IsDir()) {
				continue
			}
			file := filepath.Join(folder, entry.Name())
			if entry.IsDir() {
				err = base(file, name)
			} else {
				var expected *enginefiles.File
				if index := slices.IndexFunc(rt.Base, func(pin enginefiles.File) bool { return pin.Path == name }); index >= 0 {
					expected = &rt.Base[index]
				}
				err = add(file, "python/"+name, expected)
			}
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := base(pythonHome, ""); err != nil {
		return manifest, err
	}
	for _, file := range selected {
		pin := enginefiles.File{Size: -1, SHA256: file.SHA256}
		if err := add(file.Path, "wheels/"+filepath.Base(file.Path), &pin); err != nil {
			return manifest, err
		}
	}
	for _, pin := range sources {
		file := filepath.Join(source, "infer", filepath.FromSlash(pin.Path))
		expected := enginefiles.File{Size: pin.Size, SHA256: pin.SHA256}
		if err := add(file, "source/infer/"+pin.Path, &expected); err != nil {
			return manifest, err
		}
	}
	for _, pin := range notices {
		file := filepath.Join(source, filepath.FromSlash(pin.Path))
		if err := add(file, "source/"+pin.Path, &pin); err != nil {
			return manifest, err
		}
	}
	sort.Slice(manifest.Files, func(i, j int) bool { return manifest.Files[i].Path < manifest.Files[j].Path })
	return manifest, manifest.Validate()
}

// installInterpreter copies the base interpreter from PythonHome unless the
// component's copy already proves against the pins: the interpreter and
// its DLLs at the top, DLLs\ and Lib\ but baseExcluded. The copy is made
// beside the folder and renamed into place once its pins prove.
func (o ComponentInstall) installInterpreter() error {
	if o.verifyBase(o.BaseDir()) == nil {
		return nil
	}
	if err := o.verifyBase(o.PythonHome); err != nil {
		return fmt.Errorf("%s is not python.org's CPython %s for this platform: %w", o.PythonHome, o.runtime.PythonVersion, err)
	}
	part := o.BaseDir() + ".part"
	if err := removeTree(part); err != nil {
		return err
	}
	if err := copyBase(o.PythonHome, part, ""); err != nil {
		return err
	}
	if err := o.verifyBase(part); err != nil {
		return fmt.Errorf("the copy of the base interpreter: %w", err)
	}
	if err := removeTree(o.BaseDir()); err != nil {
		return err
	}
	return os.Rename(part, o.BaseDir())
}

// verifyBase proves the pinned base interpreter files in dir by size and
// SHA-256.
func (o ComponentInstall) verifyBase(dir string) error {
	for _, f := range o.runtime.Base {
		if err := verifyFile(filepath.Join(dir, filepath.FromSlash(f.Path)), f); err != nil {
			return err
		}
	}
	return nil
}

func verifyFile(p string, pin enginefiles.File) error {
	info, err := os.Lstat(p)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() != pin.Size {
		return fmt.Errorf("%s is not the pinned file", p)
	}
	sum, err := fileSHA256(p)
	if err != nil {
		return err
	}
	if sum != pin.SHA256 {
		return fmt.Errorf("%s is not the pinned file", p)
	}
	return nil
}

// copyBase copies from into to, rel being the slash path below the base
// interpreter's folder; a link or junction is refused.
func copyBase(from, to, rel string) error {
	if err := os.MkdirAll(to, 0o700); err != nil {
		return err
	}
	list, err := os.ReadDir(from)
	if err != nil {
		return err
	}
	for _, d := range list {
		r := d.Name()
		if rel != "" {
			r = rel + "/" + d.Name()
		}
		src, dst := filepath.Join(from, d.Name()), filepath.Join(to, d.Name())
		if d.Type()&(fs.ModeSymlink|fs.ModeIrregular) != 0 {
			return fmt.Errorf("%s is a link or a junction", src)
		}
		if !keepBase(r, d.IsDir()) {
			continue
		}
		if d.IsDir() {
			if err := copyBase(src, dst, r); err != nil {
				return err
			}
		} else {
			if err := copyFile(src, dst); err != nil {
				return err
			}
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	return err
}

// installVenv makes the venv from the copied interpreter without pip,
// unless one already stands whose launcher and pyvenv.cfg prove; a venv
// that does not prove is made again.
func (o ComponentInstall) installVenv(ctx context.Context) error {
	check := func() error {
		if err := verifyFile(o.Python(), o.runtime.Python); err != nil {
			return err
		}
		data, err := os.ReadFile(filepath.Join(o.Venv(), "pyvenv.cfg"))
		if err != nil {
			return err
		}
		home, err := checkPyvenv(data, o.runtime.PythonVersion)
		if err != nil {
			return err
		}
		if same, err := identity.SameFile(home, o.BaseDir()); err != nil || !same {
			return fmt.Errorf("the venv's home %s is not %s", home, o.BaseDir())
		}
		return nil
	}
	if check() == nil {
		return nil
	}
	if err := removeTree(o.Venv()); err != nil {
		return err
	}
	if err := run(ctx, filepath.Join(o.BaseDir(), "python.exe"), "-I", "-m", "venv", "--without-pip", o.Venv()); err != nil {
		return fmt.Errorf("make the venv: %w", err)
	}
	return check()
}

type lockedWheel struct{ Path, SHA256 string }

func lockedWheels(entries []lockEntry, folders []string) ([]lockedWheel, error) {
	wheels, err := wheelsByDigest(folders)
	if err != nil {
		return nil, err
	}
	var files []lockedWheel
	for _, e := range entries {
		i := slices.IndexFunc(e.Hashes, func(h string) bool { return wheels[h] != "" })
		if i < 0 {
			return nil, fmt.Errorf("no wheel of %s==%s with the lock's SHA-256 lies in %s", e.Name, e.Version, strings.Join(folders, ", "))
		}
		files = append(files, lockedWheel{Path: wheels[e.Hashes[i]], SHA256: e.Hashes[i]})
	}
	return files, nil
}

// installLock installs every lock entry's wheel into a fresh site-packages
// unless the venv already holds this lock, and reports whether it did.
func (o ComponentInstall) installLock(ctx context.Context, entries []lockEntry) (bool, error) {
	sum := sha256.Sum256(o.lock)
	digest := hex.EncodeToString(sum[:])
	marker := filepath.Join(o.Venv(), lockMarker)
	if held, err := os.ReadFile(marker); err == nil && strings.TrimSpace(string(held)) == digest {
		return false, nil
	}
	files, err := lockedWheels(entries, o.Wheels)
	if err != nil {
		return false, err
	}
	//unchecked: an absent marker is what a fresh venv has
	os.Remove(marker)
	site := o.sitePackages()
	if err := removeTree(site); err != nil {
		return false, err
	}
	if err := os.MkdirAll(site, 0o700); err != nil {
		return false, err
	}
	for _, whl := range files {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if err := installWheel(whl.Path, site, o.Venv()); err != nil {
			return false, err
		}
	}
	if o.Compile {
		// Compile failures are pip's to ignore too: a module that does not
		// compile is compiled again in memory when it is imported.
		//unchecked: see above
		run(ctx, o.Python(), "-I", "-m", "compileall", "-q", "-j", "0", site)
	}
	return true, os.WriteFile(marker, []byte(digest+"\n"), 0o600)
}

// run runs one Setup step's program with an environment of the system's
// own folders alone.
func run(ctx context.Context, program string, args ...string) error {
	cmd := exec.CommandContext(ctx, program, args...)
	root := os.Getenv("SystemRoot")
	cmd.Env = []string{"SystemRoot=" + root, "PATH=" + filepath.Join(root, "System32")}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s: %w: %s", filepath.Base(program), err, strings.TrimSpace(string(out)))
	}
	return nil
}

// removeTree removes a folder Setup made, after confirming it holds no link
// or junction, which a recursive removal could follow out of it.
func removeTree(dir string) error {
	if _, err := os.Lstat(dir); errors.Is(err, os.ErrNotExist) {
		return nil
	}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type()&(fs.ModeSymlink|fs.ModeIrregular) != 0 {
			return fmt.Errorf("%s is a link or a junction; Setup removes nothing below %s", p, dir)
		}
		return nil
	})
	if err != nil {
		return err
	}
	return os.RemoveAll(dir)
}

// RemoveComponent removes the PyTorch engine (TODO D5a): its sandbox
// first, which refuses while an engine runs, revokes every grant, proves
// no path still names the container and deletes the profile (KIT-S4), then
// the engine folder Setup made. The engine's state and the person's model
// files stay.
func RemoveComponent(c Component, out io.Writer) (*sandbox.Removal, error) {
	return removeComponent(c, out, sandbox.Remove, sandbox.Profiles, removeTree)
}

// The ordinary fixture supplies sandbox replies without creating profiles or
// changing grants. Production always passes the existing sandbox/tree owners.
func removeComponent(c Component, out io.Writer, removeSandbox func(string, bool) (*sandbox.Removal, error), profiles func(string) ([]string, error), removeEngine func(string) error) (*sandbox.Removal, error) {
	if err := sandbox.CheckLocalPlacement(c.Root); err != nil {
		return nil, err
	}
	matchRoot := func() error {
		if c.ExpectedRootKey == "" {
			return nil
		}
		key, err := identity.FileKey(c.Root)
		if err != nil || key != c.ExpectedRootKey {
			return errors.New("the managed PyTorch root differs from its expected identity; its data was preserved")
		}
		return nil
	}
	if err := matchRoot(); err != nil {
		return nil, err
	}
	profile, err := c.Profile()
	if err != nil {
		return nil, err
	}
	if out == nil {
		out = io.Discard
	}
	removal, err := removeSandbox(c.Manifest(), false)
	switch {
	case errors.Is(err, sandbox.ErrNotInstalled):
		if c.ExpectedRootKey == "" {
			return nil, err
		}
		// A before-install or after-removal interruption has no manifest.
		// A remaining profile would have lost its grant authority; preserve it.
		remaining, err := profiles(profile)
		if err != nil {
			return nil, err
		}
		if slices.Contains(remaining, profile) {
			return nil, errors.New("the managed PyTorch profile remains without its manifest; its data was preserved")
		}
	case err != nil:
		return removal, err
	}
	if err := matchRoot(); err != nil {
		return removal, err
	}
	if err := removeEngine(c.EngineDir()); err != nil {
		return removal, err
	}
	//unchecked: a root that still holds something else is left as it is
	rootErr := os.Remove(c.Root)
	if c.ExpectedRootKey != "" && rootErr != nil {
		return removal, fmt.Errorf("the managed PyTorch root still holds data; it was preserved: %w", rootErr)
	}
	if removal != nil {
		//unchecked: progress line; the returned error reports the outcome
		fmt.Fprintf(out, "removed the profile %s and its grants on %d paths\n", removal.Profile, len(removal.Revoked))
		for _, p := range removal.Changed {
			//unchecked: progress line
			fmt.Fprintf(out, "%s: its access list differs from the one before Setup; someone changed it since\n", p)
		}
	}
	return removal, nil
}
