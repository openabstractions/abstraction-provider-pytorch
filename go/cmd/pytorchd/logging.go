package main

import (
	"log/slog"
	"os"

	logging "github.com/openabstractions/abstraction-logging/go"
)

func resolvedDiagnostics(program string) (slog.Handler, func()) {
	return logging.ResolveDiagnostics(program, os.Stderr, logging.Resolve)
}
