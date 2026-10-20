package torch

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/windows"
)

func TestInterpreterRenameWithReadHeldStage(t *testing.T) {
	for _, test := range []string{"released", "persistent", "cancelled"} {
		t.Run(test, func(t *testing.T) {
			root := t.TempDir()
			stage, target := filepath.Join(root, "python.part"), filepath.Join(root, "python")
			if err := os.Mkdir(stage, 0o700); err != nil {
				t.Fatal(err)
			}
			const leaf = "probe.bin"
			const contents = "verified fixture bytes"
			if err := os.WriteFile(filepath.Join(stage, leaf), []byte(contents), 0o600); err != nil {
				t.Fatal(err)
			}
			name, err := windows.UTF16PtrFromString(stage)
			if err != nil {
				t.Fatal(err)
			}
			handle, err := windows.CreateFile(name, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE,
				nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS, 0)
			if err != nil {
				t.Fatal(err)
			}
			closeReader := func() {
				if handle != windows.InvalidHandle {
					if err := windows.CloseHandle(handle); err != nil {
						t.Fatal(err)
					}
					handle = windows.InvalidHandle
				}
			}
			t.Cleanup(closeReader)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			attempts := 0
			err = retryInterpreterRename(ctx, func() error {
				attempts++
				err := os.Rename(stage, target)
				if handle != windows.InvalidHandle && !interpreterRenameDenied(err) {
					t.Fatalf("read-held stage attempt %d: %v", attempts, err)
				}
				if test == "released" && attempts == 3 {
					closeReader()
				}
				if test == "cancelled" {
					cancel()
				}
				return err
			})
			switch test {
			case "released":
				if err != nil || attempts < 4 || attempts > 5 {
					t.Fatalf("promotion after release: %d attempts, %v", attempts, err)
				}
				stage, target = target, stage
			case "persistent":
				if !interpreterRenameDenied(err) || attempts != 5 {
					t.Fatalf("persistent refusal: %d attempts, %v", attempts, err)
				}
			case "cancelled":
				if !errors.Is(err, context.Canceled) || attempts != 1 {
					t.Fatalf("cancelled retry: %d attempts, %v", attempts, err)
				}
			}
			if data, err := os.ReadFile(filepath.Join(stage, leaf)); err != nil || string(data) != contents {
				t.Fatalf("final bytes: %q, %v", data, err)
			}
			if _, err := os.Lstat(target); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("unexpected second interpreter tree: %v", err)
			}
		})
	}
}
