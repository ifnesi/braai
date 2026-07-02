// Package agent implements the conversation loop that lets an Ollama model
// call read-only filesystem tools before producing a final answer.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"braai/internal/ollama"
	"braai/internal/terminal"
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
   read_files, search_name, search_content, search_semantic, stat_file, and
   (only on vision-capable models) read_image. There are no other
   capabilities. search_semantic finds files by meaning rather than exact
   text but requires the server to support embeddings — if it errors, fall
   back to search_content.
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
	// ColorLevel controls ANSI styling of the streamed output. Callers
	// should only set this when Stdout is a real terminal, since ANSI codes
	// would otherwise corrupt piped/redirected output.
	ColorLevel terminal.Level
	// Spinner, when non-nil, is stopped (and erased from the terminal) the
	// moment the first streamed token arrives. It should already be running
	// before Run is called. Only meaningful in interactive mode.
	Spinner *terminal.Spinner
	// ContextLength is the active model's context window in tokens, as
	// reported by Ollama's /api/show. When set (> 0), Run warns to
	// VerboseWriter if the conversation looks likely to approach or exceed
	// it. Zero disables the check (e.g. when the server didn't report one).
	ContextLength int
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

// SetSpinner sets the spinner that will be stopped when the first streamed
// token arrives on the next Run call. It replaces any spinner set during
// Options construction. Designed for interactive mode where a new spinner
// is created per-turn.
func (a *Agent) SetSpinner(sp *terminal.Spinner) {
	a.opts.Spinner = sp
}

// SystemMessage returns the initial system message for a new conversation.
func SystemMessage() ollama.Message {
	return ollama.Message{Role: "system", Content: systemPrompt}
}

// ToolCallRecord is a record of one tool invocation made during a Run,
// exposed so callers (e.g. --output json) can report exactly what evidence
// the model gathered, not just its final answer.
type ToolCallRecord struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
	Result    string         `json:"result,omitempty"`
	Error     string         `json:"error,omitempty"`
}

// RunResult is everything a single Run call produced.
type RunResult struct {
	Answer    string           `json:"answer"`
	Reasoning string           `json:"reasoning,omitempty"`
	ToolCalls []ToolCallRecord `json:"tool_calls,omitempty"`
	// History is the updated conversation, including this turn, for the
	// caller to pass back into the next Run call.
	History []ollama.Message `json:"-"`
}

// Run executes the tool-calling loop over the given message history (which
// must already include the system message and the latest user message) and
// returns the final result. The answer (and reasoning, if enabled) is
// streamed to opts.Stdout as it is produced, unless opts.Stdout is a
// discarding writer (e.g. for --output json, which buffers instead).
func (a *Agent) Run(ctx context.Context, history []ollama.Message) (RunResult, error) {
	toolDefs := a.registry.Definitions()
	var toolCalls []ToolCallRecord

	// Ask Ollama explicitly either way (not just omit the field when hiding)
	// so a model that defaults to always reasoning is actually told not to
	// bother computing it when the user asked to hide it.
	think := &a.opts.ShowReasoning

	warnedContext := false
	// Seed the running token estimate from the incoming history once, then
	// add each newly appended message's cost incrementally below — avoids
	// re-scanning the whole (potentially multi-turn, multi-megabyte) history
	// on every one of up to MaxToolCalls iterations.
	usedTokens := 0
	for _, m := range history {
		usedTokens += messageTokenCost(m)
	}

	for i := 0; i < a.opts.MaxToolCalls+1; i++ {
		if a.opts.ContextLength > 0 && !warnedContext {
			if ratio := float64(usedTokens) / float64(a.opts.ContextLength); ratio >= contextWarnThreshold {
				fmt.Fprintf(a.opts.VerboseWriter, "warning: conversation is ~%d%% of %s's estimated %d-token context window (~%d tokens); it may start dropping earlier context or produce degraded answers. Consider a shorter prompt, fewer read_files at once, or (in chat mode) /clear.\n",
					int(ratio*100), a.opts.Model, a.opts.ContextLength, usedTokens)
				warnedContext = true
			}
		}

		// Restart the spinner for this round trip; it was stopped by the
		// previous iteration's first streamed chunk (or never started, on
		// the first iteration, in which case Start is a caller's job before
		// calling Run). Spinner.Start is a no-op if already running, so this
		// is safe even though the caller also starts it once up front.
		if a.opts.Spinner != nil {
			a.opts.Spinner.Start()
		}
		streamer := newStreamPrinter(a.opts.Stdout, a.opts.ShowReasoning, a.opts.ColorLevel, a.opts.Spinner)

		resp, err := a.client.ChatStream(ctx, ollama.ChatRequest{
			Model:    a.opts.Model,
			Messages: history,
			Tools:    toolDefs,
			Think:    think,
		}, streamer.onChunk)
		if err != nil {
			return RunResult{History: history}, fmt.Errorf("ollama chat request failed: %w", err)
		}
		streamer.finish()

		history = append(history, resp.Message)
		usedTokens += messageTokenCost(resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			result := RunResult{
				Answer:    resp.Message.Content,
				ToolCalls: toolCalls,
				History:   history,
			}
			if a.opts.ShowReasoning {
				// Defense in depth: even though think:false is sent to
				// Ollama above, don't surface Thinking in the result if the
				// user asked to hide it, in case a model ignores that hint.
				result.Reasoning = resp.Message.Thinking
			}
			return result, nil
		}

		if a.opts.Verbose {
			fmt.Fprintf(a.opts.VerboseWriter, "[model requested %d tool call(s)]\n", len(resp.Message.ToolCalls))
		}

		if i == a.opts.MaxToolCalls {
			// Model wants to keep calling tools but we've hit the guardrail.
			return RunResult{ToolCalls: toolCalls, History: history}, fmt.Errorf("reached max tool calls (%d) without a final answer", a.opts.MaxToolCalls)
		}

		for _, tc := range resp.Message.ToolCalls {
			result, callErr := a.executeTool(ctx, tc)
			if a.opts.Verbose {
				a.logToolCall(tc, result, callErr)
			}
			content := result.Text
			record := ToolCallRecord{Name: tc.Function.Name, Arguments: tc.Function.Arguments, Result: result.Text}
			if callErr != nil {
				content = fmt.Sprintf("error: %v", callErr)
				record.Error = callErr.Error()
			}
			toolCalls = append(toolCalls, record)
			toolMsg := ollama.Message{
				Role:     "tool",
				Content:  content,
				ToolName: tc.Function.Name,
				Images:   result.Images,
			}
			history = append(history, toolMsg)
			usedTokens += messageTokenCost(toolMsg)
		}
	}

	return RunResult{ToolCalls: toolCalls, History: history}, fmt.Errorf("reached max tool calls (%d) without a final answer", a.opts.MaxToolCalls)
}

// contextWarnThreshold is the fraction of a model's context window at which
// Run starts warning that the conversation is getting large.
const contextWarnThreshold = 0.8

// charsPerToken is a rough English-text heuristic (~4 characters per token)
// used only to decide when to warn, not for anything that needs precision.
const charsPerToken = 4

// imageTokenEstimate is a rough, model-agnostic guess at how many tokens a
// single attached image consumes once encoded by a vision model. Actual
// costs vary a lot by model and image size; this exists only to keep the
// context-window warning roughly honest when read_image has been used.
const imageTokenEstimate = 768

// messageTokenCost returns a rough token estimate for a single message, so
// Run can track usedTokens incrementally (adding each new message's cost as
// it's appended) instead of re-scanning the entire history on every one of
// up to MaxToolCalls iterations. This is intentionally approximate
// (character counting, not the model's real tokenizer) — good enough to
// warn well before a conversation is likely to overflow the context window,
// not to predict it exactly.
func messageTokenCost(m ollama.Message) int {
	chars := len(m.Content) + len(m.Thinking)
	return chars/charsPerToken + len(m.Images)*imageTokenEstimate
}

// estimateContextUsage returns a rough token count for history and its ratio
// to contextLength, by summing messageTokenCost over every message. Kept as
// a convenience for computing a one-off total (e.g. seeding Run's running
// counter, or in tests) — Run itself only calls this once per invocation.
func estimateContextUsage(history []ollama.Message, contextLength int) (tokens int, ratio float64) {
	for _, m := range history {
		tokens += messageTokenCost(m)
	}
	ratio = float64(tokens) / float64(contextLength)
	return tokens, ratio
}

func (a *Agent) executeTool(ctx context.Context, tc ollama.ToolCall) (tools.Result, error) {
	return a.registry.Call(ctx, tc.Function.Name, tc.Function.Arguments)
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
