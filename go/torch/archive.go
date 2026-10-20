package torch

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-identity/go/trustedpath"
	"github.com/openabstractions/abstraction-provider-kit/go/modelfiles"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// heldModels is a set's model files as the engine reaches them: each local
// file held where it was placed and proved (KIT-M1), and each file missing
// or refused here as the archive's reference, filled into a section and
// proved before it is lent (PYTORCH-T12, storage STO-D2).
type heldModels struct {
	holds []*programfiles.Hold
	// finals is each file's held final path by its index in the set's
	// Files, "" for a file the archive serves.
	finals []string
	// refs is each file's archive reference by the same index; zero for a
	// local file.
	refs []storage.Ref
}

// Paths is each file's held final path, "" for a file the archive serves.
func (h *heldModels) Paths() []string { return h.finals }

// Release lets go of every local file held.
func (h *heldModels) Release() error {
	var failures []error
	for _, hold := range h.holds {
		if err := hold.Release(); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

// archived reports whether file i is served by the archive.
func (h *heldModels) archived(i int) bool { return i < len(h.refs) && h.refs[i].Digest != "" }

// archiveServes reports whether a set's file may come from the archive:
// the provider has one and lends sections to a sandboxed engine, the only
// way an archived file reaches the engine (never a path, never in place).
func (e TorchEngine) archiveServes() bool {
	return e.Archive != nil && e.Memory != nil && e.Sandbox != ""
}

// holdModels proves and holds sp's model files where they were placed. When
// that fails and the archive serves, each file is taken from where it was
// placed when it is there and proves, and otherwise from the archive when
// the archive finds its digest: local first, then the archive. A file found
// in neither place refuses with the placed files' own error. tell writes
// one log line per file the archive serves, naming the source and no path:
// the load does, the check before its lease does not.
func (e TorchEngine) holdModels(ctx context.Context, sp TorchSetPlace, tell bool) (*heldModels, error) {
	models, err := modelPlaced(sp)
	if err == nil {
		hold, holdErr := e.Files.LaunchPlaced(ctx, sp.Set.Name, models, sp.ModelRecord)
		if holdErr == nil {
			return &heldModels{holds: []*programfiles.Hold{hold}, finals: hold.Paths(), refs: make([]storage.Ref, len(models))}, nil
		}
		err = holdErr
	}
	if !e.archiveServes() {
		return nil, err
	}
	where := loosePlaces(sp.Places)
	h := &heldModels{finals: make([]string, len(sp.Set.Files)), refs: make([]storage.Ref, len(sp.Set.Files))}
	for i, f := range sp.Set.Files {
		if p := where[f.Name]; p != "" {
			placed := []programfiles.Placed{{Path: p, Size: f.Size, SHA256: f.SHA256, Owners: modelOwners(sp, f)}}
			hold, holdErr := e.Files.LaunchPlaced(ctx, sp.Set.Name+" "+f.Role, placed, sp.ModelRecord+"."+f.Role)
			if holdErr == nil {
				h.holds = append(h.holds, hold)
				h.finals[i] = hold.Paths()[0]
				continue
			}
		}
		// The section is counted at the pinned size and filled at the
		// archive file's size: a file of another size never fills it.
		ref, ok := e.Archive.Find("sha256:" + f.SHA256)
		if ok && ref.Size != f.Size {
			log.Printf("pytorch: %s %s: the archive copy is %d bytes, the pin %d; it is not used", sp.Set.Name, f.Role, ref.Size, f.Size)
			ok = false
		}
		if !ok {
			//unchecked: the refusal is the error returned; a failed close leaves only read-shared handles
			h.Release()
			return nil, err
		}
		h.refs[i] = ref
		if !tell {
			continue
		}
		log.Printf("pytorch: %s %s is filled from the archive", sp.Set.Name, f.Role)
	}
	return h, nil
}

// modelOwners are the owners the places record names for f, or none.
func modelOwners(sp TorchSetPlace, f VideoModelFile) trustedpath.Owners {
	if _, owners, err := readFilePlaces(sp.Places, sp.Set.Name, sp.Set.Files); err == nil {
		return owners[f.Name]
	}
	return trustedpath.Owners{}
}

// loosePlaces reads a places record's absolute paths by file name, keeping
// what it can read of a record that names only some of the set's files.
func loosePlaces(places string) map[string]string {
	out := map[string]string{}
	data, err := os.ReadFile(places)
	if err != nil {
		return out
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), "\t")
		if len(fields) == 3 && filepath.IsAbs(fields[1]) && out[fields[0]] == "" {
			out[fields[0]] = fields[1]
		}
	}
	return out
}

// refuser is a store that applies its own rule to bytes a reader refused:
// the archive moves a content-name entry aside (DL-R44).
type refuser interface {
	Refused(storage.Ref, error)
}

// lendArchived fills f's section from the archive's reference, lends it to
// the engine and returns its URI. Bytes whose digest differs are the
// archive's to set aside; the file then has no source here, and the load
// fails until it is placed or downloaded.
func (m *ModelMemory) lendArchived(t *torchProcess, f VideoModelFile, ref storage.Ref) (string, error) {
	archive := t.engine.Archive
	uri, err := m.lendFrom(t, f, archive, ref)
	if errors.Is(err, modelfiles.ErrDigest) {
		if r, ok := archive.(refuser); ok {
			r.Refused(ref, err)
		}
	}
	return uri, err
}
