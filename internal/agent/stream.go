package agent

import (
	"fmt"
	"io"
	"strings"

	"braai/internal/ollama"
	"braai/internal/terminal"
)

const (
	labelThinking     = "Thinking..."
	labelDoneThinking = "...done thinking."
	labelResponse     = "── Response ──"
)

// streamPrinter writes a model's streamed thinking/content deltas to an
// output writer as they arrive, labeling the reasoning section so it isn't
// confused with the final answer, and (when colorLevel > None) applying
// syntax-aware colour: cyan labels, dim reasoning, green code blocks, bold
// inline backtick spans.
type streamPrinter struct {
	out         io.Writer
	showReason  bool
	colorLevel  terminal.Level
	spinner     *terminal.Spinner
	inThinking  bool
	inCodeBlock bool
	printedAny  bool
	spinnerStop bool // true once we've stopped the spinner
	// printedMarker is true once the "── Response ──" divider has been
	// printed for the current turn's content. A streamPrinter is created
	// fresh per turn (per iteration of Agent.Run's loop), so this can't leak
	// across turns; see onChunk for why the divider is printed unconditionally
	// at the start of every content segment rather than only before what
	// turns out to be the final answer.
	printedMarker bool
	// trailingBacktick buffers an opening backtick if a chunk ends with one,
	// so inline spans split across chunks can still be styled on the next chunk.
	trailingBacktick string
}

func newStreamPrinter(out io.Writer, showReasoning bool, lv terminal.Level, sp *terminal.Spinner) *streamPrinter {
	return &streamPrinter{
		out:        out,
		showReason: showReasoning,
		colorLevel: lv,
		spinner:    sp,
	}
}

// stopSpinner stops the spinner on the first call; subsequent calls are no-ops.
func (p *streamPrinter) stopSpinner() {
	if !p.spinnerStop {
		p.spinnerStop = true
		if p.spinner != nil {
			p.spinner.Stop()
		}
	}
}

// onChunk is invoked once per streamed line from Ollama.
func (p *streamPrinter) onChunk(chunk ollama.ChatResponse) {
	if p.showReason && chunk.Message.Thinking != "" {
		p.stopSpinner()
		if !p.inThinking {
			// Print empty line before thinking, then label in bold+dim
			fmt.Fprintln(p.out)
			if p.colorLevel == terminal.None {
				fmt.Fprintln(p.out, labelThinking)
			} else {
				fmt.Fprint(p.out, "\x1b[1;2m") // bold + dim
				fmt.Fprintln(p.out, labelThinking)
				fmt.Fprint(p.out, "\x1b[0m") // reset
			}
			// Open a dim region that will span all reasoning chunks.
			fmt.Fprint(p.out, terminal.DimOpen(p.colorLevel))
			p.inThinking = true
		}
		fmt.Fprint(p.out, chunk.Message.Thinking)
		p.printedAny = true
	}

	if chunk.Message.Content != "" {
		p.stopSpinner()
		if p.inThinking {
			// Close the dim region opened for reasoning.
			fmt.Fprint(p.out, terminal.Reset(p.colorLevel))
			// Print done-thinking label in bold+dim
			fmt.Fprintln(p.out)
			if p.colorLevel == terminal.None {
				fmt.Fprintln(p.out, labelDoneThinking)
			} else {
				fmt.Fprint(p.out, "\x1b[1;2m") // bold + dim
				fmt.Fprintln(p.out, labelDoneThinking)
				fmt.Fprint(p.out, "\x1b[0m") // reset
			}
			fmt.Fprintln(p.out)
			p.inThinking = false
		}

		if !p.printedMarker {
			// Print empty line before content starts (only needed here — if
			// thinking ran first, its close-out above already left a blank
			// line).
			if !p.printedAny {
				fmt.Fprintln(p.out)
			}
			// Deliberately printed at the start of EVERY content segment, not
			// only the one that turns out to be the final answer: while this
			// chunk is streaming, we can't yet know whether the model will
			// also request tool calls after it (that's only known once the
			// full response is parsed). Trying to predict that would mean
			// either delaying/buffering output (losing live streaming) or
			// erasing and reprinting once we find out (fragile terminal
			// cursor math). Marking every content segment's start is simple,
			// always correct, and gives an unambiguous boundary either way —
			// if a model narrates before calling a tool and then gives a real
			// answer, you see two dividers, each correctly bounding its own
			// block, rather than one possibly-wrong guess.
			fmt.Fprintln(p.out, terminal.Bold(p.colorLevel, terminal.Cyan(p.colorLevel, labelResponse)))
			p.printedMarker = true
		}

		content := p.applyContentStyle(chunk.Message.Content)
		fmt.Fprint(p.out, content)
		p.printedAny = true
	}
}

// applyContentStyle applies per-chunk syntax colouring to answer content:
//   - toggles green colouring around fenced code blocks (``` fences)
//   - applies bold to inline `backtick` spans (including those split across chunks)
func (p *streamPrinter) applyContentStyle(s string) string {
	if p.colorLevel == terminal.None {
		return s
	}

	// Prepend any trailing backtick from the prior chunk so inline spans split
	// across chunks can still be styled.
	s = p.trailingBacktick + s
	p.trailingBacktick = ""

	var b strings.Builder
	remaining := s

	for len(remaining) > 0 {
		fence := strings.Index(remaining, "```")
		if fence < 0 {
			// No fence in this chunk — emit remainder with inline styling.
			chunk := remaining
			if p.inCodeBlock {
				b.WriteString(chunk)
			} else {
				styled := terminal.ApplyInlineCode(p.colorLevel, chunk)
				// Only carry a trailing backtick when it is genuinely an UNCLOSED
				// opening backtick: an odd number of backticks means the last one
				// is unmatched, and it must sit at the very end so ApplyInlineCode
				// left it literal (making the last byte safe to strip). A chunk
				// ending in a COMPLETED span like `code` has an even count — the
				// old check chopped the ANSI reset and emitted a stray backtick.
				if strings.Count(chunk, "`")%2 == 1 && strings.HasSuffix(chunk, "`") {
					p.trailingBacktick = "`"
					styled = styled[:len(styled)-1] // hold the unmatched backtick for next chunk
				}
				b.WriteString(styled)
			}
			break
		}

		// Text before the fence.
		before := remaining[:fence]
		if p.inCodeBlock {
			b.WriteString(before)
		} else {
			b.WriteString(terminal.ApplyInlineCode(p.colorLevel, before))
		}

		// Toggle the code-block state.
		if p.inCodeBlock {
			// Closing fence: end green region.
			b.WriteString("```")
			b.WriteString(terminal.Reset(p.colorLevel))
			p.inCodeBlock = false
		} else {
			// Opening fence: reset any prior styling, then open green region.
			b.WriteString(terminal.Reset(p.colorLevel))
			b.WriteString("\x1b[32m") // green — raw, kept open across chunks
			b.WriteString("```")
			p.inCodeBlock = true
		}

		remaining = remaining[fence+3:]
	}

	return b.String()
}

// finish tidies up trailing formatting once a stream completes, making sure
// any open styling region doesn't bleed into later output.
func (p *streamPrinter) finish() {
	// Safety net: ensure spinner is gone even if no tokens arrived.
	p.stopSpinner()

	if p.colorLevel != terminal.None {
		if p.inThinking || p.inCodeBlock {
			// Close any open dim or green region.
			fmt.Fprint(p.out, terminal.Reset(p.colorLevel))
		}
	}

	// Flush any backtick we were holding for a chunk that never came.
	if p.trailingBacktick != "" {
		fmt.Fprint(p.out, p.trailingBacktick)
		p.trailingBacktick = ""
	}

	// Always emit a trailing newline so the next prompt never appears on the
	// same line as residual output (e.g. when the model returns an empty answer).
	fmt.Fprintln(p.out)
}
