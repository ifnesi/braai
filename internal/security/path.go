// Package security provides helpers to confine filesystem access to a
// working-directory root, rejecting path traversal and symlink escapes.
package security

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Root confines filesystem operations to a directory tree.
type Root struct {
	abs string // absolute, cleaned root path
}

// NewRoot resolves dir to an absolute, cleaned path and verifies it exists
// and is a directory.
func NewRoot(dir string) (*Root, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve working dir: %w", err)
	}
	abs = filepath.Clean(abs)

	// Resolve symlinks on the root itself so later comparisons are apples-to-apples.
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return nil, fmt.Errorf("resolve working dir: %w", err)
	}

	info, err := os.Stat(resolved)
	if err != nil {
		return nil, fmt.Errorf("stat working dir: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("working dir is not a directory: %s", abs)
	}

	return &Root{abs: resolved}, nil
}

// Abs returns the absolute root path.
func (r *Root) Abs() string {
	return r.abs
}

// Resolve validates a user/model-supplied relative path against the root and
// returns the absolute path on disk. It rejects:
//   - absolute paths that escape the root
//   - ".." traversal that escapes the root
//   - symlinks whose final target resolves outside the root
//
// The returned path is safe to use directly with os functions.
func (r *Root) Resolve(relPath string) (string, error) {
	if relPath == "" || relPath == "." {
		return r.abs, nil
	}

	// Reject NUL bytes outright (invalid path component on all platforms).
	if strings.ContainsRune(relPath, 0) {
		return "", fmt.Errorf("invalid path: contains NUL byte")
	}

	// Reject absolute paths outright: tools are documented to take root-relative
	// paths, and silently reinterpreting "/etc/passwd" as root-relative would
	// be surprising even though filepath.Join would keep it contained.
	if filepath.IsAbs(relPath) {
		return "", fmt.Errorf("path must be relative to the working directory, got absolute path: %s", relPath)
	}

	joined := filepath.Join(r.abs, relPath)
	cleaned := filepath.Clean(joined)

	if !r.within(cleaned) {
		return "", fmt.Errorf("path escapes working directory: %s", relPath)
	}

	// If the path exists, resolve symlinks and re-check containment so a
	// symlink cannot be used to point outside the root.
	if resolved, err := filepath.EvalSymlinks(cleaned); err == nil {
		if !r.within(resolved) {
			return "", fmt.Errorf("path escapes working directory via symlink: %s", relPath)
		}
		return resolved, nil
	}

	// Path does not exist yet (fine for read-only tools that will just error
	// later with "not found"); still validate parent containment.
	return cleaned, nil
}

// within reports whether abs is the root itself or a descendant of it.
func (r *Root) within(abs string) bool {
	if abs == r.abs {
		return true
	}
	return strings.HasPrefix(abs, r.abs+string(filepath.Separator))
}

// RelPath returns abs expressed relative to the root, using forward-style
// cleanliness (best-effort; falls back to abs on error).
func (r *Root) RelPath(abs string) string {
	rel, err := filepath.Rel(r.abs, abs)
	if err != nil {
		return abs
	}
	return rel
}
