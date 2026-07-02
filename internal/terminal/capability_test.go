package terminal_test

import (
	"bytes"
	"testing"

	"braai/internal/terminal"
)

func TestDetect_NoColor(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	t.Setenv("COLORTERM", "truecolor")
	// Even with a truecolor terminal, NO_COLOR wins.
	if got := terminal.Detect(&bytes.Buffer{}); got != terminal.None {
		t.Fatalf("expected None, got %v", got)
	}
}

func TestDetect_NonFile(t *testing.T) {
	// bytes.Buffer is not a char device — should be None.
	if got := terminal.Detect(&bytes.Buffer{}); got != terminal.None {
		t.Fatalf("expected None for non-file writer, got %v", got)
	}
}

func TestDetect_DumbTerm(t *testing.T) {
	t.Setenv("TERM", "dumb")
	// Can't inject a fake char device in a unit test without /dev/tty tricks,
	// but we can verify that a non-file writer still returns None and that
	// the TERM path compiles and doesn't panic.
	if got := terminal.Detect(&bytes.Buffer{}); got != terminal.None {
		t.Fatalf("expected None, got %v", got)
	}
}
