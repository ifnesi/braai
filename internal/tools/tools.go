// Package tools implements the fixed, read-only filesystem toolset the agent
// may call: list_dir, read_file, search_name, search_content, stat_file.
// All tools are confined to a security.Root and never write to disk.
package tools

import (
	"braai/internal/ollama"
	"braai/internal/security"
)

// Directories that are skipped by recursive/search operations because they
// are typically large, generated, or version-control internals. This list is
// intentionally hardcoded for simplicity; adjust here if usability demands.
var skipDirNames = map[string]bool{
	".git":         true,
	"node_modules": true,
	"vendor":       true,
	".idea":        true,
	".DS_Store":    true,
}

// Registry limits controlled at construction time.
type Limits struct {
	// MaxReadBytes caps bytes returned by read_file. -1 means unlimited.
	MaxReadBytes int
	// MaxSearchFileBytes caps how many bytes of a single file search_content will scan.
	MaxSearchFileBytes int64
	// MaxSearchResults caps the number of matches search_content returns.
	MaxSearchResults int
	// MaxNameResults caps the number of matches search_name returns.
	MaxNameResults int
}

// DefaultLimits returns sane defaults; MaxReadBytes follows the CLI default of -1 (no limit).
func DefaultLimits() Limits {
	return Limits{
		MaxReadBytes:       -1,
		MaxSearchFileBytes: 2 * 1024 * 1024, // 2MB per file
		MaxSearchResults:   200,
		MaxNameResults:     500,
	}
}

// Registry executes tools against a confined root directory.
type Registry struct {
	root   *security.Root
	limits Limits
}

// NewRegistry builds a tool registry rooted at root, applying limits.
func NewRegistry(root *security.Root, limits Limits) *Registry {
	return &Registry{root: root, limits: limits}
}

// Definitions returns the Ollama tool schemas for all supported tools, in a
// stable order suitable for inclusion in a chat request.
func (r *Registry) Definitions() []ollama.Tool {
	return []ollama.Tool{
		listDirDefinition(),
		readFileDefinition(),
		searchNameDefinition(),
		searchContentDefinition(),
		statFileDefinition(),
	}
}

// Call dispatches a tool call by name with the given decoded arguments,
// returning a string result to feed back to the model.
func (r *Registry) Call(name string, args map[string]any) (string, error) {
	switch name {
	case "list_dir":
		return r.listDir(args)
	case "read_file":
		return r.readFile(args)
	case "search_name":
		return r.searchName(args)
	case "search_content":
		return r.searchContent(args)
	case "stat_file":
		return r.statFile(args)
	default:
		return "", unknownToolError(name)
	}
}
