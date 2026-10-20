//go:build !windows

package torch

import "context"

func retryInterpreterRename(_ context.Context, rename func() error) error { return rename() }
