package agent

import (
	"fmt"
	"io"

	"braai/internal/ollama"
)

// ANSI SGR codes used to visually set the reasoning trace apart from the
// final answer. "Dim" reads well on both light and dark terminal themes,
// unlike a specific color which might clash or be unreadable.
const (
	ansiDim   = "\x1b[2m"
	ansiReset = "\x1b[0m"
)

// streamPrinter writes a model's streamed thinking/content deltas to an
// output writer as they arrive, labeling the reasoning section so it isn't
// confused with the final answer, and (when useColor is set) dimming it.
type streamPrinter struct {
	out           io.Writer
	showReasoning bool
	useColor      bool
	inThinking    bool
	printedAny    bool
}

func newStreamPrinter(out io.Writer, showReasoning, useColor bool) *streamPrinter {
	return &streamPrinter{out: out, showReasoning: showReasoning, useColor: useColor}
}

// onChunk is invoked once per streamed line from Ollama.
func (p *streamPrinter) onChunk(chunk ollama.ChatResponse) {
	if p.showReasoning && chunk.Message.Thinking != "" {
		if !p.inThinking {
			if p.useColor {
				fmt.Fprint(p.out, ansiDim)
			}
			fmt.Fprintln(p.out, "--- reasoning ---")
			p.inThinking = true
		}
		fmt.Fprint(p.out, chunk.Message.Thinking)
		p.printedAny = true
	}

	if chunk.Message.Content != "" {
		if p.inThinking {
			if p.useColor {
				fmt.Fprint(p.out, ansiReset)
			}
			fmt.Fprintln(p.out, "\n--- answer ---")
			p.inThinking = false
		}
		fmt.Fprint(p.out, chunk.Message.Content)
		p.printedAny = true
	}
}

// finish tidies up trailing formatting once a stream completes, making sure
// any dim styling left open (e.g. a round that ended on a tool call right
// after reasoning, with no answer text) doesn't bleed into later output.
func (p *streamPrinter) finish() {
	if p.useColor && p.inThinking {
		fmt.Fprint(p.out, ansiReset)
	}
	if p.printedAny {
		fmt.Fprintln(p.out)
	}
}
