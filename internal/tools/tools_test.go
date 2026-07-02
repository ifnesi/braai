package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"braai/internal/security"
)

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
	return NewRegistry(root, DefaultLimits(), visionCapable)
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func TestListDir(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("list_dir", map[string]any{"path": "."})
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
	out, err := r.Call("list_dir", map[string]any{"path": ".", "extensions": []any{".md"}})
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
	r := NewRegistry(root, DefaultLimits(), false)

	out, err := r.Call("list_dir", map[string]any{"path": ".", "sort_by": "modified_time"})
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
	_, err := r.Call("read_file", map[string]any{"path": "binary.bin"})
	if err == nil {
		t.Fatal("expected error reading binary file")
	}
}

func TestReadFileReturnsContent(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("read_file", map[string]any{"path": "sub/file.txt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "hello world") {
		t.Errorf("expected content in output, got: %s", out.Text)
	}
}

func TestReadFileRejectsTraversal(t *testing.T) {
	r := setupRegistry(t, false)
	_, err := r.Call("read_file", map[string]any{"path": "../../etc/passwd"})
	if err == nil {
		t.Fatal("expected error for path traversal")
	}
}

func TestReadFilesBatch(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("read_files", map[string]any{"paths": []any{"sub/file.txt", "top.txt"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "hello world") || !strings.Contains(out.Text, "top level file") {
		t.Errorf("expected both files' contents, got: %s", out.Text)
	}
}

func TestReadFilesBatchReportsPerFileErrors(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("read_files", map[string]any{"paths": []any{"top.txt", "does-not-exist.txt"}})
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
	_, err := r.Call("read_files", map[string]any{"paths": paths})
	if err == nil {
		t.Fatal("expected error for exceeding max batch files")
	}
}

func TestReadImageRequiresVisionCapability(t *testing.T) {
	r := setupRegistry(t, false)
	_, err := r.Call("read_image", map[string]any{"path": "screenshot.png"})
	if err == nil {
		t.Fatal("expected error when model is not vision-capable")
	}
}

func TestReadImageReturnsBase64WhenVisionCapable(t *testing.T) {
	r := setupRegistry(t, true)
	out, err := r.Call("read_image", map[string]any{"path": "screenshot.png"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Images) != 1 || out.Images[0] == "" {
		t.Fatalf("expected one non-empty base64 image, got: %+v", out.Images)
	}
}

func TestReadImageRejectsNonImageExtension(t *testing.T) {
	r := setupRegistry(t, true)
	_, err := r.Call("read_image", map[string]any{"path": "top.txt"})
	if err == nil {
		t.Fatal("expected error for non-image extension")
	}
}

func TestSearchName(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("search_name", map[string]any{"pattern": "FILE"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "file.txt") {
		t.Errorf("expected file.txt in matches, got: %s", out.Text)
	}
}

func TestSearchNameExtensionFilter(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("search_name", map[string]any{"pattern": "e", "extensions": []any{".md"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "notes.md") || strings.Contains(out.Text, "top.txt") {
		t.Errorf("expected only notes.md, got: %s", out.Text)
	}
}

func TestSearchContent(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("search_content", map[string]any{"query": "kafka"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out.Text, "sub/file.txt") {
		t.Errorf("expected match in sub/file.txt, got: %s", out.Text)
	}
}

func TestStatFile(t *testing.T) {
	r := setupRegistry(t, false)
	out, err := r.Call("stat_file", map[string]any{"path": "top.txt"})
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
