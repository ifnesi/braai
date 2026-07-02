// Package agent implements the conversation loop that lets an Ollama model
// call read-only filesystem tools before producing a final answer.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"braai/internal/ollama"
	"braai/internal/tools"
)

const systemPrompt = `You are braai, a read-only filesystem research agent running locally.

Your job is to answer questions about the contents of a working directory by
using the tools provided. Rules you must follow:

1. You are strictly read-only. You cannot and must not modify, create, or
   delete any file. There is no tool for that; do not claim to have done so.
2. Never invent file contents, directory structure, or search results. If you
   have not seen something via a tool call, you do not know it.
3. Use tools whenever you need evidence to answer accurately. Prefer the
   minimum number of tool calls needed — do not call a tool "just in case" if
   you already have enough information. When you already know which several
   files you need (e.g. summarizing a batch of meeting notes), prefer
   read_files over multiple individual read_file calls.
4. Stay within the fixed toolset you are given: list_dir, read_file,
   read_files, search_name, search_content, stat_file, and (only on
   vision-capable models) read_image. There are no other capabilities.
5. When you are confident you have enough information, stop calling tools and
   give a concise, grounded final answer. Reference specific file paths when
   relevant.
6. If a tool call fails or a file cannot be found, say so plainly rather than
   guessing.
7. If read_image is available and the user asks about a screenshot, diagram,
   or photo, use it rather than guessing at an image's contents from its name.`

// Options configures a single agent run.
type Options struct {
	Model        string
	MaxToolCalls int
	Verbose      bool
	// VerboseWriter receives human-readable tracing when Verbose is true.
	VerboseWriter io.Writer
	// ShowReasoning requests the model's reasoning/thinking trace (on models
	// that support it) and streams it to Stdout as it arrives, clearly
	// separated from the final answer.
	ShowReasoning bool
	// Stdout receives the streamed answer (and reasoning, if ShowReasoning is
	// set) as it arrives from the model, token by token, rather than all at
	// once when the response completes.
	Stdout io.Writer
}

// Agent drives the chat/tool-calling loop for one conversation.
type Agent struct {
	client   *ollama.Client
	registry *tools.Registry
	opts     Options
}

// DefaultMaxToolCalls is used when Options.MaxToolCalls is left unset.
const DefaultMaxToolCalls = 100

// New builds an Agent bound to the given Ollama client and tool registry.
func New(client *ollama.Client, registry *tools.Registry, opts Options) *Agent {
	if opts.MaxToolCalls <= 0 {
		opts.MaxToolCalls = DefaultMaxToolCalls
	}
	return &Agent{client: client, registry: registry, opts: opts}
}

// SystemMessage returns the initial system message for a new conversation.
func SystemMessage() ollama.Message {
	return ollama.Message{Role: "system", Content: systemPrompt}
}

// Run executes the tool-calling loop over the given message history (which
// must already include the system message and the latest user message) and
// returns the final assistant text plus the updated history. The answer (and
// reasoning, if enabled) is streamed to opts.Stdout as it is produced.
func (a *Agent) Run(ctx context.Context, history []ollama.Message) (string, []ollama.Message, error) {
	toolDefs := a.registry.Definitions()

	var think *bool
	if a.opts.ShowReasoning {
		t := true
		think = &t
	}

	for i := 0; i < a.opts.MaxToolCalls+1; i++ {
		streamer := newStreamPrinter(a.opts.Stdout, a.opts.ShowReasoning)

		resp, err := a.client.ChatStream(ctx, ollama.ChatRequest{
			Model:    a.opts.Model,
			Messages: history,
			Tools:    toolDefs,
			Think:    think,
		}, streamer.onChunk)
		if err != nil {
			return "", history, fmt.Errorf("ollama chat request failed: %w", err)
		}
		streamer.finish()

		history = append(history, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			return resp.Message.Content, history, nil
		}

		if a.opts.Verbose {
			fmt.Fprintf(a.opts.VerboseWriter, "[model requested %d tool call(s)]\n", len(resp.Message.ToolCalls))
		}

		if i == a.opts.MaxToolCalls {
			// Model wants to keep calling tools but we've hit the guardrail.
			return "", history, fmt.Errorf("reached max tool calls (%d) without a final answer", a.opts.MaxToolCalls)
		}

		for _, tc := range resp.Message.ToolCalls {
			result, callErr := a.executeTool(tc)
			if a.opts.Verbose {
				a.logToolCall(tc, result, callErr)
			}
			content := result.Text
			if callErr != nil {
				content = fmt.Sprintf("error: %v", callErr)
			}
			history = append(history, ollama.Message{
				Role:     "tool",
				Content:  content,
				ToolName: tc.Function.Name,
				Images:   result.Images,
			})
		}
	}

	return "", history, fmt.Errorf("reached max tool calls (%d) without a final answer", a.opts.MaxToolCalls)
}

func (a *Agent) executeTool(tc ollama.ToolCall) (tools.Result, error) {
	return a.registry.Call(tc.Function.Name, tc.Function.Arguments)
}

func (a *Agent) logToolCall(tc ollama.ToolCall, result tools.Result, err error) {
	argsJSON, _ := json.Marshal(tc.Function.Arguments)
	fmt.Fprintf(a.opts.VerboseWriter, "  -> %s(%s)\n", tc.Function.Name, string(argsJSON))
	if err != nil {
		fmt.Fprintf(a.opts.VerboseWriter, "     error: %v\n", err)
		return
	}
	preview := result.Text
	const maxPreview = 500
	if len(preview) > maxPreview {
		preview = preview[:maxPreview] + "...(truncated in log)"
	}
	fmt.Fprintf(a.opts.VerboseWriter, "     result: %s\n", preview)
	if len(result.Images) > 0 {
		fmt.Fprintf(a.opts.VerboseWriter, "     attached %d image(s)\n", len(result.Images))
	}
}
