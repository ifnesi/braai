package agent

import (
	"testing"

	"braai/internal/ollama"
)

func TestBuildContextReportSplitsSystemAndConversation(t *testing.T) {
	history := []ollama.Message{
		{Role: "system", Content: "0123456789"},              // 10 chars -> 2 tokens
		{Role: "user", Content: "01234567890123456789"},      // 20 chars -> 5 tokens
		{Role: "assistant", Content: "01234567890123456789"}, // 20 chars -> 5 tokens
	}
	r := BuildContextReport("m", 1000, history, nil)

	if r.SystemPromptTokens != 2 {
		t.Fatalf("expected SystemPromptTokens=2, got %d", r.SystemPromptTokens)
	}
	if r.ConversationTokens != 10 {
		t.Fatalf("expected ConversationTokens=10, got %d", r.ConversationTokens)
	}
	// nil toolDefs marshals to the 4-byte JSON literal "null" (1 token at
	// this heuristic's 4-chars-per-token rate) rather than 0 — this asserts
	// that behavior explicitly rather than assuming an empty result.
	if r.ToolSchemaTokens != 1 {
		t.Fatalf("expected ToolSchemaTokens=1 for nil toolDefs (marshals to \"null\"), got %d", r.ToolSchemaTokens)
	}
}

func TestBuildContextReportComputesUsedAndFree(t *testing.T) {
	history := []ollama.Message{
		{Role: "system", Content: "0123456789"}, // 2 tokens
		{Role: "user", Content: "01234567"},     // 2 tokens
	}
	r := BuildContextReport("m", 100, history, nil)

	if r.Used != r.SystemPromptTokens+r.ToolSchemaTokens+r.ConversationTokens {
		t.Fatalf("Used should equal the sum of the three category fields, got Used=%d sum=%d",
			r.Used, r.SystemPromptTokens+r.ToolSchemaTokens+r.ConversationTokens)
	}
	if r.Free != 100-r.Used {
		t.Fatalf("expected Free=%d, got %d", 100-r.Used, r.Free)
	}
}

func TestBuildContextReportFreeIsZeroWhenOverBudget(t *testing.T) {
	big := make([]byte, 10000)
	for i := range big {
		big[i] = 'a'
	}
	history := []ollama.Message{{Role: "user", Content: string(big)}}
	r := BuildContextReport("m", 10, history, nil) // contextLength far below usage

	if r.Free != 0 {
		t.Fatalf("expected Free=0 when usage exceeds contextLength, got %d", r.Free)
	}
}

func TestBuildContextReportIncludesToolSchemas(t *testing.T) {
	toolDefs := []ollama.Tool{
		{Type: "function", Function: ollama.ToolFunction{Name: "read", Description: "reads a file", Parameters: map[string]any{"type": "object"}}},
	}
	withTools := BuildContextReport("m", 1000, nil, toolDefs)
	withoutTools := BuildContextReport("m", 1000, nil, nil)

	if withTools.ToolSchemaTokens <= withoutTools.ToolSchemaTokens {
		t.Fatalf("expected non-zero ToolSchemaTokens when toolDefs is non-empty: with=%d without=%d",
			withTools.ToolSchemaTokens, withoutTools.ToolSchemaTokens)
	}
}
