// Package tools implements the fixed, read-only filesystem toolset the agent
// may call: list_dir, read_file, read_files, read_image, search_name,
// search_content, stat_file. All tools are confined to a security.Root and
// never write to disk.
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

// Result is what a tool call returns to the agent: text to feed back to the
// model, plus optional images (base64-encoded, no data: URI prefix) for
// vision-capable models to inspect alongside the text.
type Result struct {
	Text   string
	Images []string
}

// textResult is a convenience constructor for the common text-only case.
func textResult(s string) Result {
	return Result{Text: s}
}

// Registry limits controlled at construction time.
type Limits struct {
	// MaxReadBytes caps bytes returned by read_file/read_files. -1 means unlimited.
	MaxReadBytes int
	// MaxSearchFileBytes caps how many bytes of a single file search_content will scan.
	MaxSearchFileBytes int64
	// MaxSearchResults caps the number of matches search_content returns.
	MaxSearchResults int
	// MaxNameResults caps the number of matches search_name returns.
	MaxNameResults int
	// MaxBatchFiles caps how many files a single read_files call may read.
	MaxBatchFiles int
	// MaxImageBytes caps the on-disk size of an image read_image will accept,
	// since the whole file is base64-encoded and sent to the model.
	MaxImageBytes int64
}

// DefaultLimits returns sane defaults; MaxReadBytes follows the CLI default of -1 (no limit).
func DefaultLimits() Limits {
	return Limits{
		MaxReadBytes:       -1,
		MaxSearchFileBytes: 2 * 1024 * 1024, // 2MB per file
		MaxSearchResults:   200,
		MaxNameResults:     500,
		MaxBatchFiles:      20,
		MaxImageBytes:      10 * 1024 * 1024, // 10MB
	}
}

// Registry executes tools against a confined root directory.
type Registry struct {
	root          *security.Root
	limits        Limits
	visionCapable bool
}

// NewRegistry builds a tool registry rooted at root, applying limits.
// visionCapable indicates whether the active Ollama model reports "vision"
// among its capabilities; read_image refuses to run when false rather than
// silently sending image data a model can't use.
func NewRegistry(root *security.Root, limits Limits, visionCapable bool) *Registry {
	return &Registry{root: root, limits: limits, visionCapable: visionCapable}
}

// Definitions returns the Ollama tool schemas for all supported tools, in a
// stable order suitable for inclusion in a chat request.
func (r *Registry) Definitions() []ollama.Tool {
	defs := []ollama.Tool{
		listDirDefinition(),
		readFileDefinition(),
		readFilesDefinition(),
		searchNameDefinition(),
		searchContentDefinition(),
		statFileDefinition(),
	}
	if r.visionCapable {
		defs = append(defs, readImageDefinition())
	}
	return defs
}

// Call dispatches a tool call by name with the given decoded arguments,
// returning the result to feed back to the model.
func (r *Registry) Call(name string, args map[string]any) (Result, error) {
	switch name {
	case "list_dir":
		return r.listDir(args)
	case "read_file":
		return r.readFile(args)
	case "read_files":
		return r.readFiles(args)
	case "read_image":
		return r.readImage(args)
	case "search_name":
		return r.searchName(args)
	case "search_content":
		return r.searchContent(args)
	case "stat_file":
		return r.statFile(args)
	default:
		return Result{}, unknownToolError(name)
	}
}
