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
// It is safe to call Stop more than once (idempotent).
type Spinner struct {
	w    io.Writer
	stop chan struct{}
	done chan struct{}
	once sync.Once // ensures Stop is idempotent
}

// NewSpinner returns a new, unstarted Spinner that writes to w.
// If lv == None the spinner is a no-op (Start and Stop become no-ops).
func NewSpinner(w io.Writer, lv Level) *Spinner {
	if lv == None {
		return &Spinner{} // nil channels → no-op
	}
	return &Spinner{
		w:    w,
		stop: make(chan struct{}),
		done: make(chan struct{}),
	}
}

// Start launches the spinner goroutine. Call Stop to erase and halt it.
func (s *Spinner) Start() {
	if s.stop == nil {
		return // no-op spinner
	}
	go s.run()
}

// Stop halts the spinner and erases it from the terminal line. Safe to call
// multiple times or before Start.
func (s *Spinner) Stop() {
	if s.stop == nil {
		return // no-op spinner
	}
	s.once.Do(func() {
		close(s.stop)
		<-s.done
		// Erase the spinner line: carriage return + spaces + carriage return.
		fmt.Fprint(s.w, "\r"+strings.Repeat(" ", 4)+"\r")
	})
}

func (s *Spinner) run() {
	defer close(s.done)
	ticker := time.NewTicker(spinnerInterval)
	defer ticker.Stop()
	i := 0
	for {
		select {
		case <-s.stop:
			return
		case <-ticker.C:
			fmt.Fprintf(s.w, "\r%s ", spinnerFrames[i%len(spinnerFrames)])
			i++
		}
	}
}
