package torch

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// PYTORCH-T6, T11: progress names work the device has finished. The real
// engine script's FLUX.2 [klein] and Hummingbird-XT generate run with a
// stand-in torch whose synchronize is recorded; every step and decode report
// follows a wait for the device. The live walkthrough of 2026-10-02 (D10) saw
// FLUX.2's four steps arrive within 0.1 s after the steps had run, and a
// video's F32 decode sit at 99 percent for 15 s.
func TestTorchEngineReportsProgressOnceTheDeviceHasDoneIt(t *testing.T) {
	python, err := basePython()
	if err != nil {
		t.Skipf("no Python for the engine script: %v", err)
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "oa_torch_engine.py")
	if err := os.WriteFile(script, torchEngineScript, 0o600); err != nil {
		t.Fatal(err)
	}
	driver, err := filepath.Abs(filepath.Join("testdata", "faketorch", "progress_settled.py"))
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(python, "-I", "-B", driver, script)
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		var stderr []byte
		if exit, ok := err.(*exec.ExitError); ok {
			stderr = exit.Stderr
		}
		t.Fatalf("driver: %v\n%s", err, stderr)
	}
	var events [][]any
	if err := json.Unmarshal(out, &events); err != nil {
		t.Fatalf("driver output %q: %v", out, err)
	}
	reports, ends := map[string]int{}, 0
	for i, e := range events {
		switch e[0] {
		case "report":
			reports[e[1].(string)]++
			if i == 0 || events[i-1][0] != "sync" {
				t.Fatalf("report %v at %d does not follow a wait for the device: %v", e, i, events)
			}
		case "end":
			// An end line follows the wait, so a device fault in the step
			// or the decode ends the log before it (KIT-P3).
			ends++
			if i == 0 || events[i-1][0] != "sync" {
				t.Fatalf("end line %v at %d does not follow a wait for the device: %v", e, i, events)
			}
		}
	}
	if ends < 9 {
		t.Fatalf("%d end lines in %v", ends, events)
	}
	// Four image steps and three video steps; one image decode and the
	// video's decode chunks.
	if reports["step"] != 7 || reports["decode"] < 2 {
		t.Fatalf("reports %v in %v", reports, events)
	}
}
