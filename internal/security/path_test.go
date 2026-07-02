package security

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func setupTestRoot(t *testing.T) (*Root, string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sub", "file.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	root, err := NewRoot(dir)
	if err != nil {
		t.Fatalf("NewRoot: %v", err)
	}
	return root, dir
}

func TestResolveWithinRoot(t *testing.T) {
	root, _ := setupTestRoot(t)

	got, err := root.Resolve("sub/file.txt")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := filepath.Join(root.Abs(), "sub", "file.txt")
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestResolveRejectsTraversal(t *testing.T) {
	root, _ := setupTestRoot(t)

	cases := []string{
		"../outside.txt",
		"sub/../../outside.txt",
		"../../etc/passwd",
	}
	for _, c := range cases {
		if _, err := root.Resolve(c); err == nil {
			t.Errorf("Resolve(%q) expected error, got nil", c)
		}
	}
}

func TestResolveRejectsAbsoluteEscape(t *testing.T) {
	root, _ := setupTestRoot(t)
	if _, err := root.Resolve("/etc/passwd"); err == nil {
		t.Errorf("expected error for absolute escape path")
	}
}

func TestResolveRejectsSymlinkEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks require elevated privileges on windows")
	}
	root, dir := setupTestRoot(t)

	outside := t.TempDir()
	outsideFile := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(outsideFile, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}

	link := filepath.Join(dir, "escape")
	if err := os.Symlink(outside, link); err != nil {
		t.Fatal(err)
	}

	if _, err := root.Resolve("escape/secret.txt"); err == nil {
		t.Errorf("expected error resolving path through escaping symlink")
	}
}

func TestResolveDot(t *testing.T) {
	root, _ := setupTestRoot(t)
	got, err := root.Resolve(".")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != root.Abs() {
		t.Errorf("got %q want %q", got, root.Abs())
	}
}

func TestResolveRejectsNulByte(t *testing.T) {
	root, _ := setupTestRoot(t)
	if _, err := root.Resolve("foo\x00bar"); err == nil {
		t.Errorf("expected error for NUL byte in path")
	}
}
