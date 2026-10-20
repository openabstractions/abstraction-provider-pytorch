package torch

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The compact FLUX.2 [klein] set's FP8 transformer and FP8/NVFP4 encoder
// become BF16 weights at load. The real engine script's unscale_weights is
// checked against weights the driver encodes from the formats' definitions,
// and refuses a scale without its weight or a format it does not know.
func TestTorchEngineUnscalesCompactWeightsToBF16(t *testing.T) {
	python, err := basePython()
	if err != nil {
		t.Skipf("no Python for the engine script: %v", err)
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "oa_torch_engine.py")
	if err := os.WriteFile(script, torchEngineScript, 0o600); err != nil {
		t.Fatal(err)
	}
	driver, err := filepath.Abs(filepath.Join("testdata", "realtorch", "unscale_reference.py"))
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(python, "-I", "-B", driver, script)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("driver: %v\n%s", err, out)
	}
	if strings.HasPrefix(string(out), "SKIP") {
		t.Skipf("the base Python has no torch: %s", out)
	}
	if strings.TrimSpace(string(out)) != `{"failures": []}` {
		t.Fatalf("driver output %s", out)
	}
}
