package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"braai/internal/security"
)

// fakeEmbedder is a minimal in-memory stand-in for *ollama.Client's Embed
// method, so search_semantic tests don't need a real Ollama server. It
// returns a deterministic vector per distinct input string and counts calls
// so tests can verify the embedding cache is actually being used.
type fakeEmbedder struct {
	calls     int
	failWith  error
	vectorFor func(input string) []float32
}

func (f *fakeEmbedder) Embed(_ context.Context, _ string, inputs []string) ([][]float32, error) {
	f.calls++
	if f.failWith != nil {
		return nil, f.failWith
	}
	out := make([][]float32, len(inputs))
	for i, in := range inputs {
		if f.vectorFor != nil {
			out[i] = f.vectorFor(in)
		} else {
			out[i] = hashVector(in)
		}
	}
	return out, nil
}

// hashVector derives a small deterministic vector from a string so identical
// or similar inputs naturally produce similar (or identical) vectors,
// without needing a real embedding model.
func hashVector(s string) []float32 {
	v := make([]float32, 8)
	for i, c := range s {
		v[i%len(v)] += float32(c)
	}
	return v
}

func setupRegistry(t *testing.T, visionCapable bool) *Registry {
	t.Helper()
	dir := t.TempDir()
	must(t, os.MkdirAll(filepath.Join(dir, "sub"), 0o755))
	must(t, os.WriteFile(filepath.Join(dir, "sub", "file.txt"), []byte("hello world\nsecond line with Kafka\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "top.txt"), []byte("top level file"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "binary.bin"), []byte{0x00, 0x01, 0x02, 'h', 'i'}, 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "notes.md"), []byte("# meeting notes"), 0o644))
	// tiny 1x1 PNG so read_image has a real file to work with.
	png := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
		0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
		0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
		0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xdd, 0x8d, 0xb0, 0x00, 0x00, 0x00,
		0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}
	must(t, os.WriteFile(filepath.Join(dir, "screenshot.png"), png, 0o644))

	root, err := security.NewRoot(dir)
	if err != nil {
		t.Fatalf("NewRoot: %v", err)
	}
	return NewRegistry(root, DefaultLimits(), visionCapable, FetchURLConfig{}, &fakeEmbedder{}, "fake-embed-model")
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func call(t *testing.T, r *Registry, name string, args map[string]any) (Result, error) {
	t.Helper()
	return r.Call(context.Background(), name, args)
}

func TestListDir(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "list_dir", map[string]any{"path": "."})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var entries []dirEntryInfo
	if err := json.Unmarshal([]byte(out.Text), &entries); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(entries) != 5 {
		t.Fatalf("expected 5 entries, got %d: %+v", len(entries), entries)
	}
}

func TestListDirExtensionFilter(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "list_dir", map[string]any{"path": ".", "extensions": []any{".md"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var entries []dirEntryInfo
	must(t, json.Unmarshal([]byte(out.Text), &entries))

	// Directories are always included regardless of the extension filter (so
	// navigation still works); only files are filtered.
	var files []string
	for _, e := range entries {
		if e.Type == "file" {
			files = append(files, e.Path)
		}
	}
	if len(files) != 1 || files[0] != "notes.md" {
		t.Fatalf("expected only notes.md among files, got: %+v (all entries: %+v)", files, entries)
	}
}

func TestListDirSortByModifiedTime(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "old.txt"), []byte("old"), 0o644))
	time.Sleep(10 * time.Millisecond)
	must(t, os.WriteFile(filepath.Join(dir, "new.txt"), []byte("new"), 0o644))

	root, err := security.NewRoot(dir)
	must(t, err)
	r := NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, &fakeEmbedder{}, "fake-embed-model")

	out, err := call(t, r, "list_dir", map[string]any{"path": ".", "sort_by": "modified_time"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var entries []dirEntryInfo
	must(t, json.Unmarshal([]byte(out.Text), &entries))
	if len(entries) != 2 || entries[0].Path != "new.txt" {
		t.Fatalf("expected new.txt first, got: %+v", entries)
	}
}

func TestReadFileRefusesBinary(t *testing.T) {
	r := setupRegistry(t, false)
	_, err := call(t, r, "read", map[string]any{"path": "binary.bin"})
	if err == nil {
		t.Fatal("expected error reading binary file")
	}
}

func TestReadFileReturnsContent(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "read", map[string]any{"path": "sub/file.txt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "hello world") {
		t.Errorf("expected content in output, got: %s", out.Text)
	}
}

func TestReadFileRejectsTraversal(t *testing.T) {
	r := setupRegistry(t, false)
	_, err := call(t, r, "read", map[string]any{"path": "../../etc/passwd"})
	if err == nil {
		t.Fatal("expected error for path traversal")
	}
}

func TestReadFilesBatch(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "read", map[string]any{"paths": []any{"sub/file.txt", "top.txt"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "hello world") || !strings.Contains(out.Text, "top level file") {
		t.Errorf("expected both files' contents, got: %s", out.Text)
	}
}

func TestReadFilesBatchReportsPerFileErrors(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "read", map[string]any{"paths": []any{"top.txt", "does-not-exist.txt"}})
	if err != nil {
		t.Fatalf("unexpected top-level error: %v", err)
	}
	if !strings.Contains(out.Text, "top level file") {
		t.Errorf("expected successful file content present, got: %s", out.Text)
	}
	if !strings.Contains(out.Text, "error:") {
		t.Errorf("expected per-file error noted, got: %s", out.Text)
	}
}

func TestReadFilesBatchRejectsTooMany(t *testing.T) {
	r := setupRegistry(t, false)
	paths := make([]any, r.limits.MaxBatchFiles+1)
	for i := range paths {
		paths[i] = "top.txt"
	}
	_, err := call(t, r, "read", map[string]any{"paths": paths})
	if err == nil {
		t.Fatal("expected error for exceeding max batch files")
	}
}

func TestReadImageRequiresVisionCapability(t *testing.T) {
	r := setupRegistry(t, false)
	_, err := call(t, r, "read_image", map[string]any{"path": "screenshot.png"})
	if err == nil {
		t.Fatal("expected error when model is not vision-capable")
	}
}

func TestReadImageReturnsBase64WhenVisionCapable(t *testing.T) {
	r := setupRegistry(t, true)
	out, err := call(t, r, "read_image", map[string]any{"path": "screenshot.png"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Images) != 1 || out.Images[0] == "" {
		t.Fatalf("expected one non-empty base64 image, got: %+v", out.Images)
	}
}

func TestReadImageRejectsNonImageExtension(t *testing.T) {
	r := setupRegistry(t, true)
	_, err := call(t, r, "read_image", map[string]any{"path": "top.txt"})
	if err == nil {
		t.Fatal("expected error for non-image extension")
	}
}

func TestSearchName(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "list_dir", map[string]any{"path": ".", "name_contains": "file", "depth": 100})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "file.txt") {
		t.Errorf("expected file.txt in matches, got: %s", out.Text)
	}
}

func TestSearchNameExtensionFilter(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "list_dir", map[string]any{"path": ".", "name_contains": "e", "extensions": []any{".md"}, "depth": 100})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "notes.md") || strings.Contains(out.Text, "top.txt") {
		t.Errorf("expected only notes.md, got: %s", out.Text)
	}
}

func TestSearchContent(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "search", map[string]any{"query": "kafka"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "sub/file.txt") {
		t.Errorf("expected match in sub/file.txt, got: %s", out.Text)
	}
}

func TestSearchContentExcerptDoesNotSplitMultiByteRune(t *testing.T) {
	dir := t.TempDir()
	// A line whose matched text starts right before byte offset 200, packed
	// with multi-byte runes (é is 2 bytes in UTF-8) so a naive s[:200] byte
	// truncation would very likely split one in half.
	line := strings.Repeat("é", 250) + " target"
	must(t, os.WriteFile(filepath.Join(dir, "unicode.txt"), []byte(line), 0o644))
	root, err := security.NewRoot(dir)
	must(t, err)
	r := NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, &fakeEmbedder{}, "fake-embed-model")

	out, err := call(t, r, "search", map[string]any{"query": "target"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !utf8.ValidString(out.Text) {
		t.Fatalf("result contains invalid UTF-8 (excerpt split a multi-byte rune): %q", out.Text)
	}
}

func TestStatFile(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := call(t, r, "stat_file", map[string]any{"path": "top.txt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var s statResult
	if err := json.Unmarshal([]byte(out.Text), &s); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if s.Type != "file" || s.SizeBytes == 0 {
		t.Errorf("unexpected stat result: %+v", s)
	}
}

func TestDefinitionsOmitReadImageWithoutVision(t *testing.T) {
	r := setupRegistry(t, false)
	for _, d := range r.Definitions() {
		if d.Function.Name == "read_image" {
			t.Fatal("read_image should not be advertised when model has no vision support")
		}
	}
}

func TestDefinitionsIncludeReadImageWithVision(t *testing.T) {
	r := setupRegistry(t, true)
	found := false
	for _, d := range r.Definitions() {
		if d.Function.Name == "read_image" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected read_image to be advertised when model has vision support")
	}
}

func TestDefinitionsAlwaysIncludeSearch(t *testing.T) {
	r := setupRegistry(t, false)
	found := false
	for _, d := range r.Definitions() {
		if d.Function.Name == "search" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected search to always be advertised")
	}
}

func TestSearchSemanticRequiresEmbedClient(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello"), 0o644))
	root, err := security.NewRoot(dir)
	must(t, err)
	r := NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, nil, "")

	_, err = call(t, r, "search", map[string]any{"query": "hello", "semantic": true})
	if err == nil {
		t.Fatal("expected error when no embed client is configured")
	}
}

func TestSearchSemanticRanksBySimilarity(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "about_cats.txt"), []byte("cats"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "about_dogs.txt"), []byte("dogs"), 0o644))
	root, err := security.NewRoot(dir)
	must(t, err)

	// A fake embedder where "cats"/"about cats" land near each other and far
	// from "dogs", so we can assert on ranking without a real model.
	fake := &fakeEmbedder{vectorFor: func(input string) []float32 {
		if strings.Contains(strings.ToLower(input), "cat") {
			return []float32{1, 0}
		}
		return []float32{0, 1}
	}}
	r := NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, fake, "fake-embed-model")

	out, err := call(t, r, "search", map[string]any{"query": "tell me about cats", "semantic": true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var parsed struct {
		Matches []semanticMatch `json:"matches"`
	}
	must(t, json.Unmarshal([]byte(out.Text), &parsed))
	if len(parsed.Matches) != 2 {
		t.Fatalf("expected 2 matches, got %d: %+v", len(parsed.Matches), parsed.Matches)
	}
	if parsed.Matches[0].Path != "about_cats.txt" {
		t.Fatalf("expected about_cats.txt ranked first, got: %+v", parsed.Matches)
	}
	if parsed.Matches[0].Score <= parsed.Matches[1].Score {
		t.Fatalf("expected top match to have a higher score: %+v", parsed.Matches)
	}
}

func TestSearchSemanticCachesEmbeddingsAcrossCalls(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello world"), 0o644))
	root, err := security.NewRoot(dir)
	must(t, err)

	fake := &fakeEmbedder{}
	r := NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, fake, "fake-embed-model")

	_, err = call(t, r, "search", map[string]any{"query": "greeting", "semantic": true})
	must(t, err)
	firstCalls := fake.calls

	_, err = call(t, r, "search", map[string]any{"query": "greeting again", "semantic": true})
	must(t, err)

	// Second call should only re-embed the query, not re-embed a.txt (whose
	// mtime hasn't changed), so total calls should grow by exactly 1.
	if fake.calls != firstCalls+1 {
		t.Fatalf("expected file embedding to be cached: first=%d second=%d", firstCalls, fake.calls)
	}
}

func TestSearchSemanticSurfacesEmbedError(t *testing.T) {
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello"), 0o644))
	root, err := security.NewRoot(dir)
	must(t, err)

	fake := &fakeEmbedder{failWith: fmt.Errorf("ollama error: This server does not support embeddings")}
	r := NewRegistry(root, DefaultLimits(), false, FetchURLConfig{}, fake, "fake-embed-model")

	_, err = call(t, r, "search", map[string]any{"query": "hello", "semantic": true})
	if err == nil || !strings.Contains(err.Error(), "does not support embeddings") {
		t.Fatalf("expected embed error to be surfaced, got: %v", err)
	}
}
