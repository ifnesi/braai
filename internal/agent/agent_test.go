package agent

import (
	"strings"
	"testing"

	"braai/internal/ollama"
)

func TestEstimateContextUsageBelowThreshold(t *testing.T) {
	history := []ollama.Message{
		{Role: "system", Content: "short system prompt"},
		{Role: "user", Content: "hello"},
	}
	tokens, ratio := estimateContextUsage(history, 100000)
	if tokens <= 0 {
		t.Fatalf("expected positive token estimate, got %d", tokens)
	}
	if ratio >= contextWarnThreshold {
		t.Fatalf("expected ratio below threshold for a tiny conversation, got %f", ratio)
	}
}

func TestEstimateContextUsageAboveThreshold(t *testing.T) {
	big := strings.Repeat("a", 100000)
	history := []ollama.Message{{Role: "user", Content: big}}
	// 100000 chars / 4 chars-per-token ≈ 25000 tokens, well over 80% of a
	// 8000-token window.
	tokens, ratio := estimateContextUsage(history, 8000)
	if tokens < 8000 {
		t.Fatalf("expected large token estimate, got %d", tokens)
	}
	if ratio < contextWarnThreshold {
		t.Fatalf("expected ratio above threshold, got %f", ratio)
	}
}

func TestEstimateContextUsageCountsImages(t *testing.T) {
	withoutImage := []ollama.Message{{Role: "tool", Content: "some text"}}
	withImage := []ollama.Message{{Role: "tool", Content: "some text", Images: []string{"base64data"}}}

	tokensWithout, _ := estimateContextUsage(withoutImage, 100000)
	tokensWith, _ := estimateContextUsage(withImage, 100000)

	if tokensWith <= tokensWithout {
		t.Fatalf("expected attaching an image to increase the token estimate: without=%d with=%d", tokensWithout, tokensWith)
	}
}
