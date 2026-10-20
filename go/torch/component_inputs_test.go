package torch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
)

func inputFile(t *testing.T, root, name, data string) enginefiles.File {
	t.Helper()
	file := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(file), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte(data))
	return enginefiles.File{Path: name, Size: int64(len(data)), SHA256: hex.EncodeToString(sum[:])}
}

func inputSnapshot(t *testing.T, root string) map[string]string {
	t.Helper()
	files := map[string]string{}
	err := filepath.WalkDir(root, func(p string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		if entry.IsDir() {
			files[name] = "directory"
		} else {
			files[name], err = fileSHA256(p)
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func TestComponentInputManifestUsesInstallationSelectorsWithoutEffects(t *testing.T) {
	engineRuns(t)
	root := t.TempDir()
	python, wheels, source := filepath.Join(root, "python"), filepath.Join(root, "wheels"), filepath.Join(root, "source")
	base := inputFile(t, python, "python.exe", "reviewed interpreter fixture")
	inputFile(t, python, "DLLs/native.pyd", "native fixture")
	inputFile(t, python, "Lib/os.py", "stdlib fixture")
	inputFile(t, python, "LICENSE.txt", "interpreter licence fixture")
	inputFile(t, python, "Lib/site-packages/personal.py", "personal install")
	inputFile(t, python, "Lib/test/private.py", "test suite")
	inputFile(t, python, "Scripts/pip.exe", "personal tool")
	wheel := inputFile(t, wheels, "nested/fixture-1.0-py3-none-any.whl", "locked wheel fixture")
	inputFile(t, wheels, "unselected-1.0-py3-none-any.whl", "unselected wheel")
	code := inputFile(t, source, "infer/models/fixture.py", "reviewed source fixture")
	notice := inputFile(t, source, "LICENSE", "reviewed source licence")
	inputFile(t, source, "README.md", "unselected source")
	rt := torchRuntime{PythonVersion: "fixture", Base: []enginefiles.File{base}}
	lock := []byte("fixture==1.0 --hash=sha256:" + wheel.SHA256 + "\n")
	sources := []TorchSourceFile{{Path: "models/fixture.py", Size: code.Size, SHA256: code.SHA256}}
	before := inputSnapshot(t, root)
	manifest, err := componentInputs(context.Background(), python, []string{wheels}, source, rt, lock, sources, []enginefiles.File{notice})
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, pin := range manifest.Files {
		names = append(names, pin.Path)
		var file string
		switch pin.Path[:7] {
		case "python/":
			file = filepath.Join(python, filepath.FromSlash(pin.Path[7:]))
		case "source/":
			file = filepath.Join(source, filepath.FromSlash(pin.Path[7:]))
		case "wheels/":
			file = filepath.Join(wheels, "nested", filepath.FromSlash(pin.Path[7:]))
		}
		if err := verifyFile(file, pin); err != nil {
			t.Fatalf("manifest does not describe current input %s: %v", pin.Path, err)
		}
	}
	want := []string{"python/DLLs/native.pyd", "python/LICENSE.txt", "python/Lib/os.py", "python/python.exe", "source/LICENSE", "source/infer/models/fixture.py", "wheels/fixture-1.0-py3-none-any.whl"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("input roster = %v, want %v", names, want)
	}
	if after := inputSnapshot(t, root); !reflect.DeepEqual(before, after) {
		t.Fatal("read-only input proof changed the filesystem")
	}
	// The copy consumer must apply exactly the same interpreter selection.
	copy := filepath.Join(root, "copy")
	if err := copyBase(python, copy, ""); err != nil {
		t.Fatal(err)
	}
	copied := inputSnapshot(t, copy)
	for _, pin := range manifest.Files {
		if len(pin.Path) > 7 && pin.Path[:7] == "python/" && copied[filepath.FromSlash(pin.Path[7:])] != pin.SHA256 {
			t.Fatalf("installation copy diverges for %s", pin.Path)
		}
	}
	if _, err := os.Stat(filepath.Join(copy, "Lib", "site-packages")); !os.IsNotExist(err) {
		t.Fatal("installation copy admitted personal site-packages")
	}
}

func TestComponentInputRefusedPinsHaveNoEffects(t *testing.T) {
	for _, invalid := range []string{"python", "wheel", "source", "notice"} {
		t.Run(invalid, func(t *testing.T) {
			root := t.TempDir()
			python, wheels, source := filepath.Join(root, "python"), filepath.Join(root, "wheels"), filepath.Join(root, "source")
			base := inputFile(t, python, "python.exe", "interpreter")
			wheel := inputFile(t, wheels, "fixture.whl", "wheel")
			code := inputFile(t, source, "infer/fixture.py", "source")
			notice := inputFile(t, source, "LICENSE", "licence")
			rt := torchRuntime{Base: []enginefiles.File{base}}
			lock := []byte("fixture==1.0 --hash=sha256:" + wheel.SHA256 + "\n")
			sources := []TorchSourceFile{{Path: "fixture.py", Size: code.Size, SHA256: code.SHA256}}
			file := map[string]string{"python": filepath.Join(python, "python.exe"), "wheel": filepath.Join(wheels, "fixture.whl"), "source": filepath.Join(source, "infer", "fixture.py"), "notice": filepath.Join(source, "LICENSE")}[invalid]
			if err := os.WriteFile(file, []byte("different bytes"), 0600); err != nil {
				t.Fatal(err)
			}
			before := inputSnapshot(t, root)
			if _, err := componentInputs(context.Background(), python, []string{wheels}, source, rt, lock, sources, []enginefiles.File{notice}); err == nil {
				t.Fatal("changed reviewed input was accepted")
			}
			if after := inputSnapshot(t, root); !reflect.DeepEqual(before, after) {
				t.Fatal("refused input proof changed the filesystem")
			}
		})
	}
}

func TestComponentInputPinnedAssemblyRefusesChangedReviewedBytes(t *testing.T) {
	for _, knownSize := range []bool{false, true} {
		root := t.TempDir()
		pin := inputFile(t, root, "fixture", "reviewed bytes")
		file := filepath.Join(root, pin.Path)
		if err := verifyFile(file, pin); err != nil {
			t.Fatal(err)
		}
		// Deterministic replacement after the selector's earlier proof, before
		// assembly. The emitted value must still match the reviewed owner pin.
		if err := os.WriteFile(file, []byte("different bytes"), 0600); err != nil {
			t.Fatal(err)
		}
		if !knownSize {
			pin.Size = -1 // wheel lock: its exact SHA-256 owns the full bytes.
		}
		before := inputSnapshot(t, root)
		if _, err := inputFilePin(file, "wheels/fixture.whl", &pin); err == nil {
			t.Fatal("manifest assembly admitted bytes changed since selection")
		}
		if after := inputSnapshot(t, root); !reflect.DeepEqual(before, after) {
			t.Fatal("refused manifest assembly changed the filesystem")
		}
	}
}
