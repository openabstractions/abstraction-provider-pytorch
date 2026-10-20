package torch

import (
	"context"
	"os"
	"slices"

	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"

	identity "github.com/openabstractions/abstraction-identity"
)

// PlaceTorchModels finds each pinned file of a PyTorch set through
// searches, proves it where it lies and records where (PYTORCH-T2, KIT-M1);
// it copies nothing.
func PlaceTorchModels(ctx context.Context, set TorchModelSet, searches []ModelSearch, places string) (map[string]string, error) {
	if err := set.valid(); err != nil {
		return nil, err
	}
	return placeFiles(ctx, set.Files, searches, places)
}

// RecordedSetFolders are the folders directly under root that a PyTorch set
// uses: one holding a file a places record in state names
// (TorchModelPlaces), or one named as a set of TorchModelSets is (KIT-M3).
func RecordedSetFolders(root, state string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	// One folder named twice is listed once, keyed by file identity
	// (identity.FileKey); a path that is not an existing folder is skipped.
	add := func(dir string) {
		if dir == "" {
			return
		}
		key, err := identity.FileKey(dir)
		if err != nil || seen[key] {
			return
		}
		if info, err := os.Lstat(dir); err == nil && info.IsDir() {
			seen[key] = true
			out = append(out, dir)
		}
	}
	for _, set := range TorchModelSets {
		places := ""
		if state != "" {
			places = TorchModelPlaces(state, set)
		}
		if err := modelplace.SetFolders(root, set.Name, places, set.Files, "pytorchd models install --set "+set.Name, add); err != nil {
			return nil, err
		}
	}
	slices.Sort(out)
	return out, nil
}
