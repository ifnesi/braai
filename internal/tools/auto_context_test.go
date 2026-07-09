package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"braai/internal/security"
)

func newAutoContextRegistry(t *testing.T, dir string, fake *fakeEmbedder) *Registry {
	t.Helper()
	root, err := security.NewRoot(dir)
	must(t, err)
	return NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, fake, "fake-embed-model")
}

func TestAutoContextRequiresEmbedClient(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello"), 0o644))
	root, err := security.NewRoot(dir)
	must(t, err)
	r := NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, nil, "")

	res, err := r.AutoContext(context.Background(), "hello", 5, 0.2, 6000, map[string]bool{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Text != "" {
		t.Fatalf("expected no-op without an embedding backend, got: %q", res.Text)
	}
}

func TestAutoContextNoOpOnEmptyDir(t *testing.T) {
	dir := t.TempDir() // no files at all
	fake := &fakeEmbedder{}
	r := newAutoContextRegistry(t, dir, fake)

	res, err := r.AutoContext(context.Background(), "anything", 5, 0.2, 6000, map[string]bool{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Text != "" {
		t.Fatalf("expected no-op on an empty directory, got: %q", res.Text)
	}
}

func TestAutoContextSkipsBelowMinScore(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("aaaa"), 0o644))

	// vectorFor makes the query and the file's chunk maximally dissimilar
	// (orthogonal vectors), so the resulting score is far below any
	// reasonable threshold.
	fake := &fakeEmbedder{vectorFor: func(input string) []float32 {
		if input == "query" {
			return []float32{1, 0}
		}
		return []float32{0, 1}
	}}
	r := newAutoContextRegistry(t, dir, fake)

	res, err := r.AutoContext(context.Background(), "query", 5, 0.2, 6000, map[string]bool{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Text != "" {
		t.Fatalf("expected no-op below minScore, got: %q", res.Text)
	}
}

func TestAutoContextInjectsRelevantChunkAndReturnsKeys(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("aaaa"), 0o644))

	// Query and file chunk share the same vector, so the score is a perfect
	// match (1.0), comfortably above minScore.
	fake := &fakeEmbedder{vectorFor: func(string) []float32 { return []float32{1, 0} }}
	r := newAutoContextRegistry(t, dir, fake)

	res, err := r.AutoContext(context.Background(), "query", 5, 0.2, 6000, map[string]bool{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Text == "" {
		t.Fatal("expected injected text for a relevant chunk")
	}
	if !strings.Contains(res.Text, "a.txt") {
		t.Fatalf("expected injected text to name the source file, got: %q", res.Text)
	}
	if len(res.Keys) != 1 || res.Keys[0] != "a.txt#1" {
		t.Fatalf("expected keys [a.txt#1], got: %v", res.Keys)
	}
}

func TestAutoContextExcludesAlreadyInjectedChunks(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("aaaa"), 0o644))

	fake := &fakeEmbedder{vectorFor: func(string) []float32 { return []float32{1, 0} }}
	r := newAutoContextRegistry(t, dir, fake)

	exclude := map[string]bool{"a.txt#1": true}
	res, err := r.AutoContext(context.Background(), "query", 5, 0.2, 6000, exclude)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Text != "" {
		t.Fatalf("expected no-op once the only candidate chunk is excluded, got: %q", res.Text)
	}
}

func TestAutoContextTruncatesAtMaxChars(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 5; i++ {
		must(t, os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%d.txt", i)), []byte(strings.Repeat("word ", 500)), 0o644))
	}
	fake := &fakeEmbedder{vectorFor: func(string) []float32 { return []float32{1, 0} }}
	r := newAutoContextRegistry(t, dir, fake)

	const maxChars = 200
	res, err := r.AutoContext(context.Background(), "query", 5, 0.2, maxChars, map[string]bool{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(res.Text, "[auto-context truncated]") {
		t.Fatalf("expected truncation note, got: %q", res.Text)
	}
}

func TestAutoContextRespectsTopK(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 5; i++ {
		must(t, os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%d.txt", i)), []byte(fmt.Sprintf("filecontent%d", i)), 0o644))
	}
	fake := &fakeEmbedder{vectorFor: func(string) []float32 { return []float32{1, 0} }}
	r := newAutoContextRegistry(t, dir, fake)

	res, err := r.AutoContext(context.Background(), "query", 2, 0.2, 6000, map[string]bool{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.Keys) != 2 {
		t.Fatalf("expected top_k=2 to cap injected chunks at 2, got %d: %v", len(res.Keys), res.Keys)
	}
}
