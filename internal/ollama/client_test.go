package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatStreamReturnsErrorWhenStreamEndsWithoutDone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate a connection that drops mid-stream: some content chunks,
		// then the server just stops writing (no done:true chunk, no error).
		fmt.Fprintln(w, `{"model":"m","message":{"role":"assistant","content":"partial"},"done":false}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, 0)
	_, err := c.ChatStream(context.Background(), ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}}, nil)
	if err == nil {
		t.Fatal("expected an error for a stream that never sent done:true, got nil")
	}
}

func TestChatStreamReturnsFullMessageOnSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"model":"m","message":{"role":"assistant","content":"hel"},"done":false}`)
		fmt.Fprintln(w, `{"model":"m","message":{"role":"assistant","content":"lo"},"done":false}`)
		fmt.Fprintln(w, `{"model":"m","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, 0)
	resp, err := c.ChatStream(context.Background(), ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Message.Content != "hello" {
		t.Fatalf("expected accumulated content %q, got %q", "hello", resp.Message.Content)
	}
	if !resp.Done {
		t.Fatal("expected Done to be true")
	}
}

func TestChatStreamSurfacesCleanErrorOnNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotImplemented)
		fmt.Fprint(w, `{"error":"This server does not support embeddings. Start it with `+"`--embeddings`"+`"}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, 0)
	_, err := c.Embed(context.Background(), "m", []string{"hi"})
	if err == nil {
		t.Fatal("expected an error")
	}
	const want = "This server does not support embeddings"
	if got := err.Error(); !strings.Contains(got, want) {
		t.Fatalf("expected error to contain %q, got %q", want, got)
	}
}
