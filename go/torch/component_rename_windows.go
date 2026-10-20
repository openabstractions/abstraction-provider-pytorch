package torch

import (
	"context"
	"errors"
	"time"

	"golang.org/x/sys/windows"
)

// Setup owns the stage under its exclusive sandbox lock throughout promotion.
func retryInterpreterRename(ctx context.Context, rename func() error) error {
	waits := [...]time.Duration{25 * time.Millisecond, 75 * time.Millisecond, 150 * time.Millisecond, 250 * time.Millisecond}
	for attempt := 0; ; attempt++ {
		err := rename()
		if attempt == len(waits) || !interpreterRenameDenied(err) {
			return err
		}
		timer := time.NewTimer(waits[attempt])
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
}

func interpreterRenameDenied(err error) bool {
	return errors.Is(err, windows.ERROR_ACCESS_DENIED) || errors.Is(err, windows.ERROR_SHARING_VIOLATION)
}
