package torch

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
)

func TestInstallRejectsNonlocalSourceBeforeCreatingEngine(t *testing.T) {
	root := filepath.Join(t.TempDir(), "component")
	err := InstallComponent(context.Background(), ComponentInstall{
		Component: Component{Root: root}, PythonHome: t.TempDir(), State: t.TempDir(),
		Wheels: []string{t.TempDir()}, Source: `\\unreachable.example\share\source`,
	})
	if !errors.Is(err, identity.ErrNotLocalPath) {
		t.Fatalf("nonlocal source: %v", err)
	}
	if _, err := os.Lstat(root); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("component was created before validation: %v", err)
	}
}

func TestRemoveWithoutManifestLeavesUnownedEngineTree(t *testing.T) {
	c := Component{Root: filepath.Join(t.TempDir(), "unowned")}
	file := filepath.Join(c.EngineDir(), "person.txt")
	if err := os.MkdirAll(filepath.Dir(file), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("person's data"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveComponent(c, nil); !errors.Is(err, sandbox.ErrNotInstalled) {
		t.Fatalf("unowned removal: %v", err)
	}
	if data, err := os.ReadFile(file); err != nil || string(data) != "person's data" {
		t.Fatalf("unowned file changed: %q (%v)", data, err)
	}
}

func TestWheelRefusesJunctionedScriptsBeforeExtraction(t *testing.T) {
	requireComponentSecurityProbeOptIn(t)
	dir := t.TempDir()
	venv := filepath.Join(dir, "venv")
	outside := filepath.Join(dir, "outside")
	if err := os.MkdirAll(venv, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outside, 0o700); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", filepath.Join(venv, "Scripts"), outside).CombinedOutput(); err != nil {
		t.Skipf("junction unavailable: %v %s", err, out)
	}
	whl, _, err := buildWheel(dir, "oa_test_pkg", "1.0", map[string]string{
		"oa_test_pkg-1.0.data/scripts/escaped.py": "print('outside')",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := installWheel(whl, filepath.Join(venv, "Lib", "site-packages"), venv); err == nil {
		t.Fatal("wheel wrote through a junctioned Scripts directory")
	}
	if _, err := os.Lstat(filepath.Join(outside, "escaped.py")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("wheel escaped its venv: %v", err)
	}
}

func TestHostCloseReleasesSandboxBeforeSetup(t *testing.T) {
	f := newTorchFixture(t)
	h := f.imageHost(t, newFakeLeases(), nil)
	reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse at dusk"))
	if reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image: %+v; log %s", reply, f.log())
	}
	h.Close()
	m, err := sandbox.ReadManifest(f.engine.Sandbox)
	if err != nil {
		t.Fatal(err)
	}
	s, err := sandbox.Begin(f.engine.Sandbox, m.Profile)
	if err != nil {
		t.Fatalf("Setup immediately after host close: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatal(err)
	}
}

func TestCopiedManifestCannotAdoptAnotherComponentProfile(t *testing.T) {
	requireComponentSecurityProbeOptIn(t)
	owner, err := torchComponent()
	if err != nil {
		t.Fatal(err)
	}
	copy := Component{Root: filepath.Join(t.TempDir(), "copy")}
	if err := os.MkdirAll(copy.Root, 0o700); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(owner.Manifest())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(copy.Manifest(), data, 0o600); err != nil {
		t.Fatal(err)
	}
	state := t.TempDir()
	if err := InstallComponent(context.Background(), ComponentInstall{Component: copy,
		PythonHome: t.TempDir(), Wheels: []string{t.TempDir()}, State: state}); err == nil {
		t.Fatal("install adopted a cloned manifest")
	}
	if _, err := os.Lstat(copy.EngineDir()); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("install wrote before checking the cloned manifest: %v", err)
	}
	engineFile := filepath.Join(copy.EngineDir(), "person.txt")
	if err := os.MkdirAll(copy.EngineDir(), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(engineFile, []byte("person"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveComponent(copy, nil); err == nil {
		t.Fatal("remove accepted a cloned manifest")
	}
	if got, err := os.ReadFile(engineFile); err != nil || string(got) != "person" {
		t.Fatalf("remove changed a foreign tree: %q (%v)", got, err)
	}
	if launch, err := (TorchEngine{Sandbox: copy.Manifest(), State: state}).prepareSandbox(nil, ""); err == nil {
		launch.Close()
		t.Fatal("launch accepted a cloned manifest")
	}
	if got, err := os.ReadFile(copy.Manifest()); err != nil || string(got) != string(data) {
		t.Fatalf("cloned manifest changed: %v", err)
	}
	m, err := sandbox.ReadManifest(owner.Manifest())
	if err != nil || m.Phase != sandbox.PhaseReady || !sandbox.ProfileExists(m.SID) {
		t.Fatalf("original component changed: %+v (%v)", m, err)
	}
}

func TestInstallRefusesJunctionedEngineBeforeEffects(t *testing.T) {
	requireComponentSecurityProbeOptIn(t)
	root := filepath.Join(t.TempDir(), "component")
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.MkdirAll(root, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outside, 0o700); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", filepath.Join(root, "engine"), outside).CombinedOutput(); err != nil {
		t.Skipf("junction unavailable: %v %s", err, out)
	}
	err := InstallComponent(context.Background(), ComponentInstall{Component: Component{Root: root},
		PythonHome: t.TempDir(), Wheels: []string{t.TempDir()}, State: t.TempDir()})
	if err == nil {
		t.Fatal("install followed the engine junction")
	}
	if entries, err := os.ReadDir(outside); err != nil || len(entries) != 0 {
		t.Fatalf("junction target was changed: %v (%v)", entries, err)
	}
	if _, err := os.Lstat(filepath.Join(root, "sandbox.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("manifest was written before refusal: %v", err)
	}
}

func TestStartedEngineTransfersOwnershipBeforeHello(t *testing.T) {
	f := newTorchFixture(t, "--no-hello=1")
	sp := TorchSetPlace{Set: f.engine.Set, Options: f.engine.Options, Places: f.engine.Places, ModelRecord: f.engine.ModelRecord}
	engine, err := f.engine.start(sp)
	if err != nil || engine == nil || engine.bootstrap == nil {
		t.Fatalf("post-spawn engine was not returned to its owner: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := engine.bootstrap(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("bootstrap did not return a readiness failure: %v", err)
	}
	engine.process.stop(stopWithin)
	if !engine.exited() {
		t.Fatal("the failed bootstrap's process or sandbox cleanup is unconfirmed")
	}
	component := Component{Root: filepath.Dir(f.engine.Sandbox)}
	profile, err := component.Profile()
	if err != nil {
		t.Fatal(err)
	}
	s, err := sandbox.Begin(f.engine.Sandbox, profile)
	if err != nil {
		t.Fatalf("Setup after the failed bootstrap: %v", err)
	}
	if err := s.Commit(); err != nil {
		t.Fatal(err)
	}
}

func TestDistinctUnicodeTwinRootsHaveDistinctProfiles(t *testing.T) {
	base := t.TempDir()
	for _, pair := range [][2]string{{"link", "linK"}, {"xs", "xſ"}} {
		var roots [2]Component
		for i, name := range pair {
			roots[i] = Component{Root: filepath.Join(base, name)}
			if err := os.Mkdir(roots[i].Root, 0o700); err != nil {
				t.Fatal(err)
			}
		}
		first, err := identity.FileKey(roots[0].Root)
		if err != nil {
			t.Fatal(err)
		}
		second, err := identity.FileKey(roots[1].Root)
		if err != nil {
			t.Fatal(err)
		}
		if first == second {
			t.Skipf("this filesystem aliases %q and %q", pair[0], pair[1])
		}
		p, err := roots[0].Profile()
		if err != nil {
			t.Fatal(err)
		}
		q, err := roots[1].Profile()
		if err != nil {
			t.Fatal(err)
		}
		if p == q {
			t.Fatalf("distinct roots share %s", p)
		}
		if len(p) != 64 || len(q) != 64 {
			t.Fatalf("custom profiles do not use the full 160-bit root digest: %q, %q", p, q)
		}
	}
}
