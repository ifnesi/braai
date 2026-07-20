package agent

import (
	"encoding/json"

	"braai/internal/ollama"
)

// ContextReport is a rough, estimated breakdown of what's consuming a
// conversation's context window, for the /context command. All fields use
// the same char-count heuristic as messageTokenCost (see its doc comment) —
// this is intentionally an estimate, not a real tokenizer count, exactly
// like the existing 80%-full warning in Run.
type ContextReport struct {
	Model string
	// ContextLength is the model's total context window in tokens, as
	// reported by Ollama's /api/show. Zero if the server didn't report one
	// (e.g. a model Ollama has no metadata for), in which case Used/Ratio
	// are still computed but Free/UsedRatio are not meaningful.
	ContextLength int

	SystemPromptTokens int // the fixed agent.SystemMessage()
	ToolSchemaTokens   int // registry.Definitions(), resent on every request
	ConversationTokens int // every other message in history (user/assistant/tool)

	// Used is the sum of the three fields above. Free is ContextLength-Used
	// (zero if ContextLength is 0 or usage already exceeds it).
	Used int
	Free int
}

// BuildContextReport estimates token usage for a conversation without
// making any Ollama request. history should be the full conversation
// (system message included); toolDefs should be registry.Definitions() —
// passed in rather than fetched here so the caller controls exactly which
// registry/model state the report reflects.
func BuildContextReport(model string, contextLength int, history []ollama.Message, toolDefs []ollama.Tool) ContextReport {
	r := ContextReport{Model: model, ContextLength: contextLength}

	for _, m := range history {
		cost := messageTokenCost(m)
		if m.Role == "system" {
			r.SystemPromptTokens += cost
		} else {
			r.ConversationTokens += cost
		}
	}

	// Tool schemas aren't ollama.Message values, so they don't go through
	// messageTokenCost; estimate their marshaled JSON size with the same
	// chars-per-token heuristic, since that's what's actually serialized
	// into the request body on every call.
	if b, err := json.Marshal(toolDefs); err == nil {
		r.ToolSchemaTokens = len(b) / charsPerToken
	}

	r.Used = r.SystemPromptTokens + r.ToolSchemaTokens + r.ConversationTokens
	if r.ContextLength > r.Used {
		r.Free = r.ContextLength - r.Used
	}
	return r
}
