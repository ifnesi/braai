package terminal_test

import (
	"bytes"
	"testing"
	"time"

	"braai/internal/terminal"
)

func TestSpinnerStopBeforeStartDoesNotDeadlock(t *testing.T) {
	sp := terminal.NewSpinner(&bytes.Buffer{}, terminal.Basic)

	done := make(chan struct{})
	go func() {
		sp.Stop()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Stop() before Start() deadlocked")
	}
}

func TestSpinnerStopIsIdempotent(t *testing.T) {
	sp := terminal.NewSpinner(&bytes.Buffer{}, terminal.Basic)
	sp.Start()
	sp.Stop()
	sp.Stop() // must not panic or block
}

func TestSpinnerRestartsAfterStop(t *testing.T) {
	var buf bytes.Buffer
	sp := terminal.NewSpinner(&buf, terminal.Basic)

	sp.Start()
	time.Sleep(150 * time.Millisecond) // let at least one frame render
	sp.Stop()
	firstLen := buf.Len()
	if firstLen == 0 {
		t.Fatal("expected spinner to write at least one frame before first Stop")
	}

	// A second Start/Stop cycle should work too (e.g. one per tool-calling
	// round trip), not be a no-op because the spinner was already used once.
	sp.Start()
	time.Sleep(150 * time.Millisecond)
	sp.Stop()
	if buf.Len() <= firstLen {
		t.Fatal("expected spinner to write more output on a second Start/Stop cycle")
	}
}

func TestSpinnerNoneLevelIsNoop(t *testing.T) {
	var buf bytes.Buffer
	sp := terminal.NewSpinner(&buf, terminal.None)
	sp.Start()
	time.Sleep(150 * time.Millisecond)
	sp.Stop()
	if buf.Len() != 0 {
		t.Fatalf("expected no output for a None-level spinner, got %q", buf.String())
	}
}
