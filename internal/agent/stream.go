package agent

import (
	"fmt"
	"io"
	"strings"

	"braai/internal/ollama"
	"braai/internal/terminal"
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
			// Print empty line before thinking, then "Thinking..." in bold+dim
			fmt.Fprintln(p.out)
			if p.colorLevel == terminal.None {
				fmt.Fprintln(p.out, "Thinking...")
			} else {
				fmt.Fprint(p.out, "\x1b[1;2m") // bold + dim
				fmt.Fprintln(p.out, "Thinking...")
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
			// Print "...done thinking." in bold+dim
			fmt.Fprintln(p.out)
			if p.colorLevel == terminal.None {
				fmt.Fprintln(p.out, "...done thinking.")
			} else {
				fmt.Fprint(p.out, "\x1b[1;2m") // bold + dim
				fmt.Fprintln(p.out, "...done thinking.")
				fmt.Fprint(p.out, "\x1b[0m") // reset
			}
			fmt.Fprintln(p.out)
			p.inThinking = false
		}

		// Print empty line before content starts
		if !p.printedAny {
			fmt.Fprintln(p.out)
		}

		content := p.applyContentStyle(chunk.Message.Content)
		fmt.Fprint(p.out, content)
		p.printedAny = true
	}
}

// applyContentStyle applies per-chunk syntax colouring to answer content:
//   - toggles green colouring around fenced code blocks (``` fences)
//   - applies bold to inline `backtick` spans
func (p *streamPrinter) applyContentStyle(s string) string {
	if p.colorLevel == terminal.None {
		return s
	}

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
				b.WriteString(terminal.ApplyInlineCode(p.colorLevel, chunk))
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
	if p.printedAny {
		fmt.Fprintln(p.out)
	}
}
