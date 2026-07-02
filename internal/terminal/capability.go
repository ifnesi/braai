// Package terminal provides helpers for detecting terminal colour capabilities
// and applying ANSI styling in a level-aware way.
package terminal

import (
	"io"
	"os"
)

// Level represents the ANSI colour capability of an output stream.
type Level int

const (
	// None means no ANSI codes should be emitted (pipe, file, NO_COLOR, dumb).
	None Level = iota
	// Basic means standard 8/16-colour SGR codes are safe to use.
	Basic
	// Full means the terminal advertises 256-colour or true-colour support.
	Full
)

// Detect inspects environment variables and the nature of w to decide the
// highest colour level that is safe to use when writing to w.
//
// Priority order:
//  1. NO_COLOR set (any value) → None
//  2. w is not a real terminal (char device) → None
//  3. TERM=dumb → None
//  4. COLORTERM=truecolor or COLORTERM=24bit → Full
//  5. Otherwise → Basic
func Detect(w io.Writer) Level {
	// Honour https://no-color.org/
	if _, set := os.LookupEnv("NO_COLOR"); set {
		return None
	}

	// Only emit codes when writing to an actual terminal.
	if !IsTerminal(w) {
		return None
	}

	if os.Getenv("TERM") == "dumb" {
		return None
	}

	switch os.Getenv("COLORTERM") {
	case "truecolor", "24bit":
		return Full
	}

	return Basic
}

// IsTerminal reports whether v (expected to be an *os.File, but accepted as
// any so it can check both io.Reader stdin and io.Writer stdout without two
// near-identical functions) is backed by a real terminal character device.
// Non-*os.File values (pipes represented as bytes.Buffer in tests, wrapped
// writers, etc.) are treated as non-terminals — the safe default, since
// callers use this to decide whether it's safe to emit ANSI codes or slurp
// piped input.
func IsTerminal(v any) bool {
	f, ok := v.(*os.File)
	if !ok {
		return false
	}
	stat, err := f.Stat()
	if err != nil {
		return false
	}
	return (stat.Mode() & os.ModeCharDevice) != 0
}
