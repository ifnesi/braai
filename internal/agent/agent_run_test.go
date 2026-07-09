package agent

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"braai/internal/ollama"
	"braai/internal/security"
	"braai/internal/tools"
)

func newTestAgent(t *testing.T, serverURL string, showReasoning bool) *Agent {
	t.Helper()
	dir := t.TempDir()
	root, err := security.NewRoot(dir)
	if err != nil {
		t.Fatalf("NewRoot: %v", err)
	}
	registry := tools.NewRegistry(root, tools.DefaultLimits(), false, tools.FetchURLConfig{}, nil, "")
	client := ollama.NewClient(serverURL, 0)
	return New(client, registry, Options{
		Model:         "m",
		Stdout:        io.Discard,
		ShowReasoning: showReasoning,
	})
}

// TestRunStripsThinkingFromHistory confirms that once a turn completes, the
// message actually appended to History has its Thinking field cleared —
// even though the turn's own RunResult.Reasoning still reports it — so a
// later turn's request doesn't re-send this turn's full chain-of-thought.
// This is what keeps follow-up questions from getting progressively slower
// purely from Ollama having to reprocess ever-larger accumulated reasoning
// traces that no longer serve any purpose once already shown.
func TestRunStripsThinkingFromHistory(t *testing.T) {
	const longThinking = "step one, step two, step three, this is a long chain of thought"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"model":"m","message":{"role":"assistant","thinking":%q,"content":"the answer"},"done":true,"done_reason":"stop"}`+"\n", longThinking)
	}))
	defer srv.Close()

	a := newTestAgent(t, srv.URL, true)
	history := []ollama.Message{{Role: "user", Content: "hi"}}

	result, err := a.Run(context.Background(), history)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Reasoning != longThinking {
		t.Fatalf("expected RunResult.Reasoning to still report this turn's thinking, got %q", result.Reasoning)
	}

	last := result.History[len(result.History)-1]
	if last.Role != "assistant" {
		t.Fatalf("expected last history message to be the assistant's, got role %q", last.Role)
	}
	if last.Thinking != "" {
		t.Fatalf("expected Thinking to be stripped from the historical message, got %q", last.Thinking)
	}
	if last.Content != "the answer" {
		t.Fatalf("expected Content to be preserved, got %q", last.Content)
	}
}

// TestRunStripsThinkingRegardlessOfShowReasoning confirms the stripping isn't
// conditional on --hide-reasoning: even when reasoning display is off for
// this run, any Thinking a model produces anyway must still not accumulate
// in history (belt-and-suspenders, since the request already asks the model
// not to think in this mode).
func TestRunStripsThinkingRegardlessOfShowReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"model":"m","message":{"role":"assistant","thinking":"leaked thinking","content":"the answer"},"done":true,"done_reason":"stop"}`)
	}))
	defer srv.Close()

	a := newTestAgent(t, srv.URL, false) // ShowReasoning: false, i.e. --hide-reasoning
	history := []ollama.Message{{Role: "user", Content: "hi"}}

	result, err := a.Run(context.Background(), history)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	last := result.History[len(result.History)-1]
	if last.Thinking != "" {
		t.Fatalf("expected Thinking to be stripped from history even with ShowReasoning=false, got %q", last.Thinking)
	}
}

// TestRunHistoryDoesNotAccumulateThinkingAcrossTurns simulates a two-turn
// conversation and confirms the second request sent to Ollama contains only
// the first turn's Content, not its Thinking — i.e. the stripping actually
// prevents accumulation across turns, not just within one.
func TestRunHistoryDoesNotAccumulateThinkingAcrossTurns(t *testing.T) {
	var secondRequestBody string
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 2 {
			body, _ := io.ReadAll(r.Body)
			secondRequestBody = string(body)
		}
		fmt.Fprintf(w, `{"model":"m","message":{"role":"assistant","thinking":"turn %d thinking","content":"turn %d answer"},"done":true,"done_reason":"stop"}`+"\n", callCount, callCount)
	}))
	defer srv.Close()

	a := newTestAgent(t, srv.URL, true)
	history := []ollama.Message{{Role: "user", Content: "first question"}}

	result, err := a.Run(context.Background(), history)
	if err != nil {
		t.Fatalf("unexpected error on first turn: %v", err)
	}

	history = append(result.History, ollama.Message{Role: "user", Content: "second question"})
	if _, err := a.Run(context.Background(), history); err != nil {
		t.Fatalf("unexpected error on second turn: %v", err)
	}

	if secondRequestBody == "" {
		t.Fatal("did not capture the second request body")
	}
	if contains := (func(s, substr string) bool {
		for i := 0; i+len(substr) <= len(s); i++ {
			if s[i:i+len(substr)] == substr {
				return true
			}
		}
		return false
	}); contains(secondRequestBody, "turn 1 thinking") {
		t.Fatalf("second request re-sent the first turn's thinking trace, it should have been stripped: %s", secondRequestBody)
	}
}
