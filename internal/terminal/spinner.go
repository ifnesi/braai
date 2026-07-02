package terminal

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

const spinnerInterval = 100 * time.Millisecond

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Spinner shows an animated braille-dot spinner on a single terminal line.
// It supports repeated Start/Stop cycles (e.g. once per tool-calling round
// trip) and is safe to call Stop before Start, or more than once, in either
// order — both are no-ops rather than a deadlock.
type Spinner struct {
	w  io.Writer
	lv Level

	mu      sync.Mutex
	running bool
	stop    chan struct{}
	done    chan struct{}
}

// NewSpinner returns a new, unstarted Spinner that writes to w.
// If lv == None the spinner is a no-op (Start and Stop become no-ops).
func NewSpinner(w io.Writer, lv Level) *Spinner {
	return &Spinner{w: w, lv: lv}
}

// Start launches the spinner goroutine if it isn't already running. Safe to
// call repeatedly (e.g. once per tool-calling round trip): a Start after a
// prior Stop begins a fresh animation cycle.
func (s *Spinner) Start() {
	if s.lv == None {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		return
	}
	s.running = true
	s.stop = make(chan struct{})
	s.done = make(chan struct{})
	go s.run(s.stop, s.done)
}

// Stop halts the spinner and erases it from the terminal line. Safe to call
// when the spinner was never started, or is already stopped — both are
// no-ops rather than blocking forever waiting for a goroutine that was
// never launched.
func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	stopCh, doneCh := s.stop, s.done
	s.running = false
	s.mu.Unlock()

	close(stopCh)
	<-doneCh
	// Erase the spinner line: carriage return + spaces + carriage return.
	fmt.Fprint(s.w, "\r"+strings.Repeat(" ", 4)+"\r")
}

func (s *Spinner) run(stop, done chan struct{}) {
	defer close(done)
	ticker := time.NewTicker(spinnerInterval)
	defer ticker.Stop()
	i := 0
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			fmt.Fprintf(s.w, "\r%s ", spinnerFrames[i%len(spinnerFrames)])
			i++
		}
	}
}
