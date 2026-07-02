package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"braai/internal/security"
)

func setupRegistry(t *testing.T) *Registry {
	t.Helper()
	dir := t.TempDir()
	must(t, os.MkdirAll(filepath.Join(dir, "sub"), 0o755))
	must(t, os.WriteFile(filepath.Join(dir, "sub", "file.txt"), []byte("hello world\nsecond line with Kafka\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "top.txt"), []byte("top level file"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "binary.bin"), []byte{0x00, 0x01, 0x02, 'h', 'i'}, 0o644))

	root, err := security.NewRoot(dir)
	if err != nil {
		t.Fatalf("NewRoot: %v", err)
	}
	return NewRegistry(root, DefaultLimits())
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func TestListDir(t *testing.T) {
	r := setupRegistry(t)
	out, err := r.Call("list_dir", map[string]any{"path": "."})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var entries []dirEntryInfo
	if err := json.Unmarshal([]byte(out), &entries); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d: %+v", len(entries), entries)
	}
}

func TestReadFileRefusesBinary(t *testing.T) {
	r := setupRegistry(t)
	_, err := r.Call("read_file", map[string]any{"path": "binary.bin"})
	if err == nil {
		t.Fatal("expected error reading binary file")
	}
}

func TestReadFileReturnsContent(t *testing.T) {
	r := setupRegistry(t)
	out, err := r.Call("read_file", map[string]any{"path": "sub/file.txt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "hello world") {
		t.Errorf("expected content in output, got: %s", out)
	}
}

func TestReadFileRejectsTraversal(t *testing.T) {
	r := setupRegistry(t)
	_, err := r.Call("read_file", map[string]any{"path": "../../etc/passwd"})
	if err == nil {
		t.Fatal("expected error for path traversal")
	}
}

func TestSearchName(t *testing.T) {
	r := setupRegistry(t)
	out, err := r.Call("search_name", map[string]any{"pattern": "FILE"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "file.txt") {
		t.Errorf("expected file.txt in matches, got: %s", out)
	}
}

func TestSearchContent(t *testing.T) {
	r := setupRegistry(t)
	out, err := r.Call("search_content", map[string]any{"query": "kafka"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "sub/file.txt") {
		t.Errorf("expected match in sub/file.txt, got: %s", out)
	}
}

func TestStatFile(t *testing.T) {
	r := setupRegistry(t)
	out, err := r.Call("stat_file", map[string]any{"path": "top.txt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var s statResult
	if err := json.Unmarshal([]byte(out), &s); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if s.Type != "file" || s.SizeBytes == 0 {
		t.Errorf("unexpected stat result: %+v", s)
	}
}
