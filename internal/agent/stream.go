package agent

import (
	"fmt"
	"io"

	"braai/internal/ollama"
)

// streamPrinter writes a model's streamed thinking/content deltas to an
// output writer as they arrive, labeling the reasoning section so it isn't
// confused with the final answer.
type streamPrinter struct {
	out           io.Writer
	showReasoning bool
	inThinking    bool
	printedAny    bool
}

func newStreamPrinter(out io.Writer, showReasoning bool) *streamPrinter {
	return &streamPrinter{out: out, showReasoning: showReasoning}
}

// onChunk is invoked once per streamed line from Ollama.
func (p *streamPrinter) onChunk(chunk ollama.ChatResponse) {
	if p.showReasoning && chunk.Message.Thinking != "" {
		if !p.inThinking {
			fmt.Fprintln(p.out, "--- reasoning ---")
			p.inThinking = true
		}
		fmt.Fprint(p.out, chunk.Message.Thinking)
		p.printedAny = true
	}

	if chunk.Message.Content != "" {
		if p.inThinking {
			fmt.Fprintln(p.out, "\n--- answer ---")
			p.inThinking = false
		}
		fmt.Fprint(p.out, chunk.Message.Content)
		p.printedAny = true
	}
}

// finish tidies up trailing formatting once a stream completes.
func (p *streamPrinter) finish() {
	if p.printedAny {
		fmt.Fprintln(p.out)
	}
}
