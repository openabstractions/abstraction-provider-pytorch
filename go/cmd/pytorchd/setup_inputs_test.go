package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
)

func TestSetupInputsDispatchIsReadOnlyAndReturnsOwnerManifest(t *testing.T) {
	old := componentInputRoster
	t.Cleanup(func() { componentInputRoster = old })
	calls := 0
	want := enginefiles.Manifest{Files: []enginefiles.File{{Path: "python/python.exe", Size: 3, SHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}}
	componentInputRoster = func(ctx context.Context, python string, wheels []string, source string) (enginefiles.Manifest, error) {
		calls++
		if python != "input-python" || !reflect.DeepEqual(wheels, []string{"input-wheels"}) || source != "input-source" {
			t.Fatal("dispatch changed the explicit package input roots")
		}
		return want, nil
	}
	var out, failure bytes.Buffer
	if code := runSetup([]string{"inputs", "--python-home", "input-python", "--wheels", "input-wheels", "--source", "input-source"}, &out, &failure); code != 0 {
		t.Fatalf("inputs = %d: %s", code, failure.String())
	}
	var got enginefiles.Manifest
	if err := json.Unmarshal(out.Bytes(), &got); err != nil || !reflect.DeepEqual(got, want) || calls != 1 {
		t.Fatalf("owner manifest = %v, calls %d, error %v", got, calls, err)
	}
	root := filepath.Join(t.TempDir(), "uncreated-component")
	for _, flags := range [][]string{{"--root", root}, {"--state", root}, {"--no-compile"}, {"--root="}, {"--state="}, {"--no-compile=false"}, {"--expected-root-key", "key"}, {"--expected-root-key="}} {
		out.Reset()
		failure.Reset()
		if code := runSetup(append([]string{"inputs"}, flags...), &out, &failure); code != 2 {
			t.Fatalf("component flags were accepted by inputs: %d, %s", code, failure.String())
		}
	}
	if calls != 1 {
		t.Fatal("refused component flags reached input enumeration")
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatal("input dispatch created a component/state root")
	}
	componentInputRoster = func(context.Context, string, []string, string) (enginefiles.Manifest, error) {
		return enginefiles.Manifest{}, errors.New("input pin changed")
	}
	out.Reset()
	failure.Reset()
	if code := runSetup([]string{"inputs"}, &out, &failure); code != 1 || out.Len() != 0 {
		t.Fatalf("refused pins emitted a manifest: %d, %s", code, out.String())
	}
}
