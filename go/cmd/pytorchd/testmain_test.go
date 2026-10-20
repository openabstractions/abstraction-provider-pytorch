package main

import (
	"os"
	"testing"

	"github.com/openabstractions/abstraction-provider-pytorch/go/internal/personfolders"
)

// TestMain runs this package's tests with the person's per-user folders
// inside a temporary folder (personfolders.Main).
func TestMain(m *testing.M) { os.Exit(personfolders.Main(m)) }
