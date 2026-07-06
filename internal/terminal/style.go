package terminal

import (
	"fmt"
	"strings"
)

// ANSI SGR escape sequences used throughout the package.
// Only applied when Level >= Basic.
const (
	ansiBold   = "\x1b[1m"
	ansiDim    = "\x1b[2m"
	ansiRed    = "\x1b[31m"
	ansiGreen  = "\x1b[32m"
	ansiYellow = "\x1b[33m"
	ansiBlue   = "\x1b[34m"
	ansiCyan   = "\x1b[36m"
	ansiReset  = "\x1b[0m"
)

// Bold wraps s with bold SGR codes when lv >= Basic, otherwise returns s as-is.
func Bold(lv Level, s string) string {
	if lv == None {
		return s
	}
	return ansiBold + s + ansiReset
}

// Dim wraps s with dim SGR codes when lv >= Basic, otherwise returns s as-is.
func Dim(lv Level, s string) string {
	if lv == None {
		return s
	}
	return ansiDim + s + ansiReset
}

// Red wraps s with red foreground SGR codes when lv >= Basic.
func Red(lv Level, s string) string {
	if lv == None {
		return s
	}
	return ansiRed + s + ansiReset
}

// Green wraps s with green foreground SGR codes when lv >= Basic.
func Green(lv Level, s string) string {
	if lv == None {
		return s
	}
	return ansiGreen + s + ansiReset
}

// Cyan wraps s with cyan foreground SGR codes when lv >= Basic.
func Cyan(lv Level, s string) string {
	if lv == None {
		return s
	}
	return ansiCyan + s + ansiReset
}

// Yellow wraps s with yellow foreground SGR codes when lv >= Basic.
func Yellow(lv Level, s string) string {
	if lv == None {
		return s
	}
	return ansiYellow + s + ansiReset
}

// Blue wraps s with blue foreground SGR codes when lv >= Basic.
// Used for the prompt in light mode where cyan can be hard to read.
func Blue(lv Level, s string) string {
	if lv == None {
		return s
	}
	return ansiBlue + s + ansiReset
}

// DimOpen returns the raw open-dim escape when lv >= Basic (empty string
// otherwise). Used by streamPrinter to start a dim region that spans many
// individual Write calls.
func DimOpen(lv Level) string {
	if lv == None {
		return ""
	}
	return ansiDim
}

// Reset returns the SGR reset escape when lv >= Basic (empty string
// otherwise). Used to close an open styling region.
func Reset(lv Level) string {
	if lv == None {
		return ""
	}
	return ansiReset
}

// ApplyInlineCode wraps every inline backtick span (` … `) in s with bold
// SGR when lv >= Basic. Only single-backtick spans that open and close on
// the same chunk are matched — multi-line spans are left untouched.
func ApplyInlineCode(lv Level, s string) string {
	if lv == None {
		return s
	}
	// Fast-path: no backtick at all.
	if !strings.ContainsRune(s, '`') {
		return s
	}
	var b strings.Builder
	remaining := s
	for {
		open := strings.Index(remaining, "`")
		if open < 0 {
			b.WriteString(remaining)
			break
		}
		close := strings.Index(remaining[open+1:], "`")
		if close < 0 {
			// No matching close — leave the rest as-is.
			b.WriteString(remaining)
			break
		}
		close += open + 1 // absolute index of closing backtick

		b.WriteString(remaining[:open])
		b.WriteString(ansiBold)
		b.WriteString(remaining[open : close+1]) // includes both backticks
		b.WriteString(ansiReset)
		remaining = remaining[close+1:]
	}
	return b.String()
}

// Link wraps text as an OSC 8 hyperlink pointing to url when lv >= Basic,
// so terminals that support the protocol (iTerm2, VS Code, macOS Terminal ≥
// Sonoma, most modern Linux terminals) render it as a clickable link.
// When lv == None (piped/redirected output, NO_COLOR, dumb terminal) the
// raw text is returned unchanged — no escape sequences are emitted.
//
// Use file:// URLs for local paths; append #L<n> for a line-number hint
// that VS Code and some other editors will jump to:
//
//	terminal.Link(lv, "main.go:42", "file:///abs/path/main.go#L42")
func Link(lv Level, text, url string) string {
	if lv == None {
		return text
	}
	// OSC 8 protocol: ESC ] 8 ; params ; uri ST  visible-text  ESC ] 8 ; ; ST
	// ST = BEL (\a) is used here for broad compatibility.
	return fmt.Sprintf("\x1b]8;;%s\a%s\x1b]8;;\a", url, text)
}
