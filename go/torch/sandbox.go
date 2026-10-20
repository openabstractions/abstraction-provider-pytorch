package torch

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-provider-kit/go/refusal"
	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
)

// The PyTorch engine runs inside the provider kit's sandbox: a Windows
// AppContainer with no capabilities, which is its network containment
// (CONTRACT.md PYTORCH-T12, provider kit KIT-S1..KIT-S6). Setup applies its
// grants once; each launch checks them and starts inside it.

// EngineSandbox is a launch the engine's sandbox refused: not installed,
// being installed, updated or removed, or missing a grant (KIT-S3).
const EngineSandbox = refusal.EngineSandbox

// SandboxProfile is the PyTorch engine kind's profile,
// oa.engine.pytorch.torch, with label after it for a second installation
// (KIT-S1).
func SandboxProfile(label string) (string, error) {
	return sandbox.ProfileName("pytorch", "torch", label)
}

// SandboxGrants are the grants Setup applies for the engine (PYTORCH-T12):
// read and execute on the component's engine folder (the interpreter, the
// venv and AMD's source when Setup placed it there) and on the engine's
// own script folder in its state; read, write and delete on the caches,
// the temporary and the working folder every cache variable points into;
// read and execute on a source folder outside the engine folder. Model
// files get no grant: each launch lends them (KIT-S6).
func SandboxGrants(engineDir, state, source string) []sandbox.Grant {
	grants := []sandbox.Grant{
		{Path: engineDir, Access: sandbox.ReadExecute},
		{Path: filepath.Join(state, "engine"), Access: sandbox.ReadExecute},
		{Path: filepath.Join(state, "cache"), Access: sandbox.Modify},
		{Path: filepath.Join(state, "temp"), Access: sandbox.Modify},
		{Path: filepath.Join(state, "work"), Access: sandbox.Modify},
	}
	if source != "" {
		grants = append(grants, sandbox.Grant{Path: source, Access: sandbox.ReadExecute})
	}
	return grants
}

// sandboxFolders are the folders of the engine's state SandboxGrants
// names, which Setup makes before it grants them.
func sandboxFolders(state string) []string {
	return []string{filepath.Join(state, "engine"), filepath.Join(state, "cache"), filepath.Join(state, "temp"),
		filepath.Join(state, "work")}
}

// sandboxNeeds is what one launch needs of the sandbox: read and execute on
// every engine file it holds, by final path (the launcher, pyvenv.cfg, the
// base interpreter's files, the script, the key tables and AMD's source),
// and on the venv's site-packages; read, write and delete on the caches,
// the temporary and the working folder.
func (e TorchEngine) sandboxNeeds(finals []string, site string) []sandbox.Grant {
	needs := make([]sandbox.Grant, 0, len(finals)+4)
	for _, f := range finals {
		needs = append(needs, sandbox.Grant{Path: f, Access: sandbox.ReadExecute})
	}
	needs = append(needs, sandbox.Grant{Path: site, Access: sandbox.ReadExecute})
	for _, d := range []string{"cache", "temp", "work"} {
		needs = append(needs, sandbox.Grant{Path: filepath.Join(e.State, d), Access: sandbox.Modify})
	}
	return needs
}

// errNoSandbox is a Windows engine configured without its sandbox: it is
// refused rather than started uncontained.
var errNoSandbox = errors.New("the PyTorch engine runs only inside its sandbox on Windows; install it with pytorchd setup install")

// needsSandbox reports whether this platform runs the engine contained:
// Windows, the one platform with a pinned stack and AppContainers.
func needsSandbox() bool { return runtime.GOOS == "windows" }

// prepareSandbox checks the engine's sandbox for a launch holding finals,
// whose venv's site-packages is site, and returns the prepared launch the
// engine starts in; nil where no sandbox is configured (no pinned platform
// but Windows, which requires one). A refusal is EngineSandbox with the
// reason.
func (e TorchEngine) prepareSandbox(finals []string, site string) (*sandbox.Launch, error) {
	if e.Sandbox == "" {
		if needsSandbox() {
			return nil, refuse(EngineSandbox, "%v", errNoSandbox)
		}
		return nil, nil
	}
	component := Component{Root: filepath.Dir(e.Sandbox)}
	if !identity.SamePathName(e.Sandbox, component.Manifest()) {
		return nil, refuse(EngineSandbox, "the sandbox manifest must be %s", component.Manifest())
	}
	if _, err := component.Profile(); err != nil {
		return nil, refuse(EngineSandbox, "%v", err)
	}
	for _, d := range []string{"cache", "temp", "work"} {
		if err := sandbox.CheckLocalPlacement(filepath.Join(e.State, d)); err != nil {
			return nil, refuse(Unavailable, "the engine's %s folder: %v", d, err)
		}
	}
	l, err := sandbox.Prepare(e.Sandbox, e.sandboxNeeds(finals, site))
	if err != nil {
		return nil, refuse(EngineSandbox, "%v", err)
	}
	return l, nil
}

// lend gives the running engine a read-only handle to the held file final
// (KIT-S6): the provider opens it beside its own hold, which keeps the
// bytes the proved ones, and the engine reads them through the handle.
func (t *torchProcess) lend(final string) (uintptr, error) {
	f, err := os.Open(final)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return t.process.owned.LendTo(f, t.conn.pid)
}

// modelFiles is each model file of sp as the engine receives it: its
// pinned name, size and SHA-256, and, in the sandbox, an fd: URI naming a
// read-only handle lent to the engine (VISION 2026-10-01, "model bytes");
// outside it, the held file's final path.
func (t *torchProcess) modelFiles(sp TorchSetPlace, finals []string) (map[string]any, error) {
	if len(finals) != len(sp.Set.Files) {
		return nil, errors.New("the held files do not match the proof")
	}
	out := map[string]any{}
	for i, f := range sp.Set.Files {
		entry := map[string]any{"name": f.Name, "size": f.Size, "sha256": f.SHA256}
		if t.contained {
			v, err := t.lend(finals[i])
			if err != nil {
				return nil, fmt.Errorf("lend %s to the engine: %w", f.Name, err)
			}
			entry["uri"] = fmt.Sprintf("fd:%d?offset=0&length=%d&sha256=%s", v, f.Size, f.SHA256)
		} else {
			entry["path"] = finals[i]
		}
		out[f.Role] = entry
	}
	return out, nil
}
