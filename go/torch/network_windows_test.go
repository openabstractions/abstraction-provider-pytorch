package torch

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-provider-kit/go/testsupport"
)

// sentinel is what the test's own control connection and datagram carry,
// so the witness tells them from anything the engine sent.
const sentinel = "oa-network-sentinel"

// networkWitness is a loopback TCP listener and UDP receiver the test
// holds: the authority on what reached them. Each arrival that is not the
// test's own sentinel is counted against the engine, with the probe's name
// it carries.
type networkWitness struct {
	tcp      net.Listener
	udp      net.PacketConn
	mu       sync.Mutex
	foreign  []string
	tcpMarks chan struct{}
	udpMarks chan struct{}
}

func newNetworkWitness(t *testing.T) *networkWitness {
	t.Helper()
	tcp, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	udp, err := net.ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		tcp.Close()
		t.Fatal(err)
	}
	w := &networkWitness{tcp: tcp, udp: udp, tcpMarks: make(chan struct{}, 4), udpMarks: make(chan struct{}, 4)}
	t.Cleanup(func() {
		tcp.Close()
		udp.Close()
	})
	go func() {
		for {
			c, err := tcp.Accept()
			if err != nil {
				return
			}
			_ = c.SetReadDeadline(time.Now().Add(500 * time.Millisecond))
			b, _ := io.ReadAll(io.LimitReader(c, int64(len(sentinel))))
			c.Close()
			if string(b) == sentinel {
				w.tcpMarks <- struct{}{}
			} else {
				w.note(fmt.Sprintf("tcp %q from %s", b, c.RemoteAddr()))
			}
		}
	}()
	go func() {
		buf := make([]byte, 512)
		for {
			n, from, err := udp.ReadFrom(buf)
			if err != nil {
				return
			}
			if string(buf[:n]) == sentinel {
				w.udpMarks <- struct{}{}
			} else {
				w.note(fmt.Sprintf("udp %q from %s", buf[:n], from))
			}
		}
	}()
	return w
}

func (w *networkWitness) note(s string) {
	w.mu.Lock()
	w.foreign = append(w.foreign, s)
	w.mu.Unlock()
}

// mark sends the test's own sentinel to both and waits for each to arrive.
// The first mark is the uncontained control; a later one flushes every
// earlier arrival through the witness, since each queue keeps its order.
func (w *networkWitness) mark(t *testing.T) {
	t.Helper()
	c, err := net.DialTimeout("tcp", w.tcp.Addr().String(), 2*time.Second)
	if err != nil {
		t.Fatalf("uncontained loopback TCP: %v", err)
	}
	_, err = c.Write([]byte(sentinel))
	c.Close()
	if err != nil {
		t.Fatal(err)
	}
	u, err := net.Dial("udp", w.udp.LocalAddr().String())
	if err != nil {
		t.Fatalf("uncontained loopback UDP: %v", err)
	}
	_, err = u.Write([]byte(sentinel))
	u.Close()
	if err != nil {
		t.Fatal(err)
	}
	for name, marks := range map[string]chan struct{}{"tcp": w.tcpMarks, "udp": w.udpMarks} {
		select {
		case <-marks:
		case <-time.After(5 * time.Second):
			t.Fatalf("the %s witness never saw the test's own sentinel", name)
		}
	}
}

func (w *networkWitness) arrivals() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]string(nil), w.foreign...)
}

// PYTORCH-T12: the engine as pytorchd launches it, the reviewed venv's
// interpreter inside the component's capability-less AppContainer, reaches
// no network. On loopback, which the audit hook allows, Python's TCP and
// UDP sockets, a TCP connect through Winsock called natively as torch's or
// ROCm's native code would, and a child interpreter's TCP and UDP without
// the hook all fail to reach the test's listener and receiver; each probe
// names itself in what it sends, so an arrival says which one got out.
// With OA_PROVIDER_KIT_OUTSIDE_ADDRESS set, the native and the child's TCP
// connects to an address off the machine fail too, after the test itself
// reached it.
func TestContainedEngineReachesNoNetwork(t *testing.T) {
	engineRuns(t)
	w := newNetworkWitness(t)
	w.mark(t)
	args := []string{"--net-tcp=" + w.tcp.Addr().String(), "--net-udp=" + w.udp.LocalAddr().String()}
	outside := testsupport.OutsideNetwork(t)
	if outside != "" {
		args = append(args, "--net-outside="+outside)
	}
	f := newTorchFixture(t, args...)
	if f.engine.Sandbox == "" {
		t.Fatal("the fixture's engine has no sandbox")
	}
	h := f.imageHost(t, newFakeLeases(), nil)
	reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse at dusk"))
	if reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image: %+v; log %s", reply, f.log())
	}
	h.Close()
	w.mark(t)
	if got := w.arrivals(); len(got) > 0 {
		t.Errorf("the contained engine reached the test's loopback witness: %v", got)
	}
	var probe map[string]any
	for _, r := range f.records(t) {
		if r["event"] == "network" {
			probe = r
		}
	}
	if probe == nil {
		t.Fatalf("the engine ran no network probe; records %v; log %s", f.records(t), f.log())
	}
	child, ok := probe["child"].(map[string]any)
	if !ok || child["failed"] != nil {
		t.Fatalf("the engine's child interpreter did not run its probe: %v", probe["child"])
	}
	// TCP says itself whether it connected. A UDP send reports success
	// whether or not the datagram leaves; the receiver above is its
	// authority, and here each must at least have been attempted.
	for name, v := range map[string]any{"python-tcp": probe["python_loopback_tcp"], "native-tcp": probe["native_loopback_tcp"],
		"child-tcp": child["loopback_tcp"]} {
		if s, ok := v.(string); !ok || !strings.HasPrefix(s, "blocked") {
			t.Errorf("%s: %v, want blocked", name, v)
		}
	}
	for name, v := range map[string]any{"python-udp": probe["python_loopback_udp"], "child-udp": child["loopback_udp"]} {
		if s, ok := v.(string); !ok || s != "sent" && !strings.HasPrefix(s, "blocked") {
			t.Errorf("%s: %v, want sent or blocked", name, v)
		}
	}
	if outside != "" {
		for name, v := range map[string]any{"native-outside-tcp": probe["native_outside_tcp"], "child-outside-tcp": child["outside_tcp"]} {
			if s, ok := v.(string); !ok || !strings.HasPrefix(s, "blocked") {
				t.Errorf("%s: %v, want blocked", name, v)
			}
		}
	} else if probe["native_outside_tcp"] != nil || child["outside_tcp"] != nil {
		t.Errorf("an outside probe ran without %s: %v", testsupport.OutsideAddressEnv, probe)
	}
	if t.Failed() {
		t.Logf("probe %v", probe)
	}
}

// installWheelComponent installs a component, with Setup's byte-compile,
// whose lock is the one wheel of files, and removes it when the test ends,
// leaving no profile (KIT-S4).
func installWheelComponent(t *testing.T, files map[string]string) (Component, torchRuntime, error) {
	t.Helper()
	base, err := basePython()
	if err != nil {
		t.Skipf("no Python for the component: %v", err)
	}
	home := filepath.Dir(base)
	rt, err := pinsOf(home)
	if err != nil {
		t.Fatal(err)
	}
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	_, sum, err := buildWheel(filepath.Join(dir, "wheels"), "oa_planted_pkg", "1.0", files, nil)
	if err != nil {
		t.Fatal(err)
	}
	c := Component{Root: filepath.Join(dir, "component")}
	err = InstallComponent(context.Background(), ComponentInstall{Component: c, PythonHome: home,
		Wheels: []string{filepath.Join(dir, "wheels")}, State: filepath.Join(dir, "state", "torch"), Compile: true,
		runtime: rt, lock: lockFor([3]string{"oa-planted-pkg", "1.0", sum})})
	t.Cleanup(func() {
		removal, err := RemoveComponent(c, nil)
		if err != nil {
			t.Errorf("remove the component: %v", err)
		} else if len(removal.Changed) > 0 || !removal.ProfileDeleted {
			t.Errorf("the component's removal %+v", removal)
		}
	})
	return c, rt, err
}

// Setup's byte-compile runs no wheel code (InstallComponent: "nothing in a
// wheel runs"): the interpreter outside the sandbox starts without its site
// module, so a wheel's .pth file, which site executes, never runs
// uncontained during Setup, where it could reach the network. The control
// shows the same .pth runs once site does.
func TestSetupCompileRunsNoWheelCode(t *testing.T) {
	engineRuns(t)
	marker := filepath.Join(t.TempDir(), "pth-ran")
	c, rt, err := installWheelComponent(t, map[string]string{
		"oa_planted_pkg/__init__.py": "VALUE = 7\n",
		"oa_planted_pkg.pth":         "import os; open(" + pyString(marker) + ", 'w').close()\n",
	})
	if err != nil {
		t.Fatal(err)
	}
	site := filepath.Join(c.Venv(), filepath.FromSlash(rt.SitePackages))
	if entries, err := os.ReadDir(filepath.Join(site, "oa_planted_pkg", "__pycache__")); err != nil || len(entries) == 0 {
		t.Fatalf("Setup compiled nothing: %v %v", entries, err)
	}
	if _, err := os.Lstat(marker); err == nil {
		t.Fatal("a wheel's .pth file ran during Setup's compile, outside the sandbox")
	}
	if out, err := exec.Command(c.Python(), "-I", "-c", "pass").CombinedOutput(); err != nil {
		t.Fatalf("control: %v %s", err, out)
	}
	if _, err := os.Lstat(marker); err != nil {
		t.Fatalf("control: the .pth file does not run even with site: %v", err)
	}
}

// A wheel cannot replace the venv's launcher or its pyvenv.cfg, which names
// the interpreter the launcher starts: Setup refuses the wheel, and no
// program a wheel carried runs outside the sandbox. The planted launcher is
// this test binary, which records beside itself that it ran.
func TestWheelCannotReplaceTheVenvInterpreter(t *testing.T) {
	engineRuns(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	planted, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, entry, data string }{
		{"launcher", "oa_planted_pkg-1.0.data/scripts/python.exe", string(planted)},
		{"pyvenv.cfg", "oa_planted_pkg-1.0.data/data/pyvenv.cfg", "home = C:\\planted\ninclude-system-site-packages = false\nversion = 3.12.10\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, _, err := installWheelComponent(t, map[string]string{"oa_planted_pkg/__init__.py": "VALUE = 7\n", tc.entry: tc.data})
			if err == nil || !strings.Contains(err.Error(), "would replace the venv's") {
				t.Errorf("Setup installed a wheel that replaces the venv's %s: %v", tc.name, err)
			}
			if _, err := os.Lstat(filepath.Join(filepath.Dir(c.Python()), plantedRan)); err == nil {
				t.Errorf("the wheel's own launcher ran outside the sandbox")
			}
		})
	}
}

// pyString is s as a single-quoted Python string literal.
func pyString(s string) string {
	return "'" + strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), "'", `\'`) + "'"
}
