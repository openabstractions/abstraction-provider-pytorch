package torch

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
)

// buildWheel writes a wheel of dist at version holding files (slash path to
// text) to dir, with its METADATA, WHEEL and a RECORD naming every file
// with its SHA-256 and size; it returns the wheel's path and SHA-256.
// extra entries are written beside without a RECORD row.
func buildWheel(dir, dist, version string, files map[string]string, extra map[string]string) (string, string, error) {
	info := dist + "-" + version + ".dist-info"
	all := map[string]string{
		info + "/METADATA": "Metadata-Version: 2.1\nName: " + dist + "\nVersion: " + version + "\n",
		info + "/WHEEL":    "Wheel-Version: 1.0\nGenerator: oa-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
	}
	for k, v := range files {
		all[k] = v
	}
	names := make([]string, 0, len(all))
	for k := range all {
		names = append(names, k)
	}
	slices.Sort(names)
	var record strings.Builder
	for _, n := range names {
		sum := sha256.Sum256([]byte(all[n]))
		fmt.Fprintf(&record, "%s,sha256=%s,%d\n", n, base64.RawURLEncoding.EncodeToString(sum[:]), len(all[n]))
	}
	record.WriteString(info + "/RECORD,,\n")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", "", err
	}
	p := filepath.Join(dir, dist+"-"+version+"-py3-none-any.whl")
	f, err := os.Create(p)
	if err != nil {
		return "", "", err
	}
	z := zip.NewWriter(f)
	write := func(name, data string) error {
		w, err := z.Create(name)
		if err == nil {
			_, err = w.Write([]byte(data))
		}
		return err
	}
	for _, n := range names {
		if err := write(n, all[n]); err != nil {
			return "", "", err
		}
	}
	for n, data := range extra {
		if err := write(n, data); err != nil {
			return "", "", err
		}
	}
	if err := write(info+"/RECORD", record.String()); err != nil {
		return "", "", err
	}
	if err := z.Close(); err != nil {
		return "", "", err
	}
	if err := f.Close(); err != nil {
		return "", "", err
	}
	sum, err := fileSHA256(p)
	return p, sum, err
}

// lockFor is a lock naming each wheel by its SHA-256.
func lockFor(entries ...[3]string) []byte {
	var b strings.Builder
	for _, e := range entries {
		fmt.Fprintf(&b, "%s==%s \\\n    --hash=sha256:%s\n    # test\n", e[0], e[1], e[2])
	}
	return []byte(b.String())
}

// pinsOf pins the CPython installed at home as the reviewed stack pins
// python.org's: the files that run and the venv launcher its venv module
// copies.
func pinsOf(home string) (torchRuntime, error) {
	pin := func(p string) (enginefiles.File, error) {
		data, err := os.ReadFile(p)
		if err != nil {
			return enginefiles.File{}, err
		}
		sum := sha256.Sum256(data)
		return enginefiles.File{Path: filepath.Base(p), Size: int64(len(data)), SHA256: hex.EncodeToString(sum[:])}, nil
	}
	var base []enginefiles.File
	for _, pattern := range []string{"python.exe", "python3*.dll", "vcruntime140*.dll", "DLLs/_socket.pyd", "DLLs/select.pyd"} {
		matches, _ := filepath.Glob(filepath.Join(home, filepath.FromSlash(pattern)))
		for _, m := range matches {
			f, err := pin(m)
			if err != nil {
				return torchRuntime{}, err
			}
			rel, _ := filepath.Rel(home, m)
			f.Path = filepath.ToSlash(rel)
			base = append(base, f)
		}
	}
	launcher, err := pin(filepath.Join(home, "Lib", "venv", "scripts", "nt", "python.exe"))
	if err != nil {
		return torchRuntime{}, err
	}
	launcher.Path = "python.exe"
	out, err := exec.Command(filepath.Join(home, "python.exe"), "-I", "-c", "import platform; print(platform.python_version())").Output()
	if err != nil {
		return torchRuntime{}, err
	}
	return torchRuntime{Release: "test", Python: launcher, Base: base, PythonVersion: strings.TrimSpace(string(out)),
		SitePackages: "Lib/site-packages", TargetFamily: "gfx-test"}, nil
}

// testComponent is the engine as Setup installs it, once per test binary,
// in a temporary folder of its own: a copy of this machine's CPython, a
// venv holding one small wheel, and a sandbox whose profile no other
// installation shares. Beside the component's grants, its work folder is
// granted read, write and delete, and every fixture makes its state, AMD
// source stand-in, engine scripts and records there; model files lie
// outside it and reach the engine as lent handles.
type testComponent struct {
	Component
	runtime torchRuntime
	work    string
}

var (
	torchTestRoot   string
	torchTestOnce   sync.Once
	torchTestResult *testComponent
	torchTestErr    error
)

// torchComponent is the binary's test component on Windows; elsewhere no
// sandbox exists and it is nil.
func torchComponent() (*testComponent, error) {
	torchTestOnce.Do(func() { torchTestResult, torchTestErr = installTestComponent() })
	return torchTestResult, torchTestErr
}

func installTestComponent() (*testComponent, error) {
	if runtime.GOOS != "windows" {
		return nil, nil
	}
	base, err := basePython()
	if err != nil {
		return nil, err
	}
	home := filepath.Dir(base)
	rt, err := pinsOf(home)
	if err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp("", "pytorch-component-")
	if err != nil {
		return nil, err
	}
	if dir, err = filepath.EvalSymlinks(dir); err != nil {
		return nil, err
	}
	torchTestRoot = dir
	whl, sum, err := buildWheel(filepath.Join(dir, "wheels"), "oa_test_pkg", "1.0",
		map[string]string{"oa_test_pkg/__init__.py": "VALUE = 7\n"}, nil)
	if err != nil {
		return nil, err
	}
	_ = whl
	c := &testComponent{Component: Component{Root: filepath.Join(dir, "component")}, runtime: rt, work: filepath.Join(dir, "work")}
	if err := os.MkdirAll(c.work, 0o700); err != nil {
		return nil, err
	}
	err = InstallComponent(context.Background(), ComponentInstall{Component: c.Component, PythonHome: home,
		Wheels: []string{filepath.Join(dir, "wheels")}, State: filepath.Join(c.work, "state", "torch"),
		runtime: rt, lock: lockFor([3]string{"oa-test-pkg", "1.0", sum})})
	if err != nil {
		return nil, err
	}
	profile, err := c.Profile()
	if err != nil {
		return nil, err
	}
	s, err := sandbox.Begin(c.Manifest(), profile)
	if err != nil {
		return nil, err
	}
	if err := s.Grant(sandbox.Grant{Path: c.work, Access: sandbox.Modify}); err != nil {
		s.Close()
		return nil, err
	}
	return c, s.Commit()
}

// removeTestComponent removes the test component and reports what of it
// was left: a grant, an access list that differs, or the profile.
func removeTestComponent() error {
	if torchTestResult == nil {
		if torchTestRoot != "" {
			return os.RemoveAll(torchTestRoot)
		}
		return nil
	}
	c := torchTestResult
	m, err := sandbox.ReadManifest(c.Manifest())
	if err != nil {
		return err
	}
	removal, err := RemoveComponent(c.Component, nil)
	if err != nil {
		return err
	}
	if len(removal.Changed) > 0 || !removal.ProfileDeleted {
		return fmt.Errorf("the test component's removal %+v", removal)
	}
	if holders, err := sandbox.Holders(torchTestRoot, m.SID); err != nil || len(holders) > 0 {
		return fmt.Errorf("paths still naming the test container: %v %v", holders, err)
	}
	return os.RemoveAll(torchTestRoot)
}
