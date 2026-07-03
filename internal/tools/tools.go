// Package tools implements the fixed, read-only filesystem toolset the agent
// may call: list_dir, read_file, read_files, read_image, search_name,
// search_content, search_semantic, stat_file. All tools are confined to a
// security.Root and never write to disk.
package tools

import (
	"context"

	"braai/internal/ollama"
	"braai/internal/security"
	"braai/internal/textextract"
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
	// MaxSearchFileBytes caps how many bytes of a single file search_content or
	// search_semantic will read.
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
	// MaxSemanticFiles caps how many files a single search_semantic call will
	// embed and compare, to bound cost/latency on large trees.
	MaxSemanticFiles int
	// MaxSemanticResults caps how many ranked matches search_semantic returns.
	MaxSemanticResults int
	// MaxEmbedChars caps how much of a file's text is sent for embedding.
	MaxEmbedChars int
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
		MaxSemanticFiles:   200,
		MaxSemanticResults: 10,
		MaxEmbedChars:      8000,
	}
}

// embedder is the minimal interface search_semantic needs from an Ollama
// client, kept small so tests can substitute a fake instead of hitting a
// real server. *ollama.Client satisfies this via its Embed method.
type embedder interface {
	Embed(ctx context.Context, model string, inputs []string) ([][]float32, error)
}

// Registry executes tools against a confined root directory.
type Registry struct {
	root          *security.Root
	limits        Limits
	visionCapable bool

	embedClient embedder
	embedModel  string
	// embedCache holds one vector per file path, keyed by path, and is
	// reused across search_semantic calls within the process's lifetime as
	// long as the file's mtime hasn't changed — an all-in-memory,
	// brute-force cache (no persistence, no vector DB).
	embedCache map[string]embedCacheEntry

	// documentChunkCache caches extracted/chunked documents by relative path.
	// Set by read_document and search_document, read by get_chunk.
	documentChunkCache map[string][]textextract.Chunk

	// chunkEmbedder embeds and semantically ranks document chunks for
	// search_document. It is long-lived so its per-document embedding cache
	// is reused across calls. Nil when no embedding client is configured.
	chunkEmbedder *textextract.ChunkEmbedder
}

type embedCacheEntry struct {
	modTime int64
	vector  []float32
}

// NewRegistry builds a tool registry rooted at root, applying limits.
// visionCapable indicates whether the active Ollama model reports "vision"
// among its capabilities; read_image refuses to run when false rather than
// silently sending image data a model can't use. embedClient/embedModel are
// used by search_semantic; embedClient may be nil if the caller has no
// Ollama client available, in which case search_semantic reports a clear
// error instead of panicking.
func NewRegistry(root *security.Root, limits Limits, visionCapable bool, embedClient embedder, embedModel string) *Registry {
	r := &Registry{
		root:                   root,
		limits:                 limits,
		visionCapable:          visionCapable,
		embedClient:            embedClient,
		embedModel:             embedModel,
		embedCache:             make(map[string]embedCacheEntry),
		documentChunkCache:     make(map[string][]textextract.Chunk),
	}
	if embedClient != nil {
		r.chunkEmbedder = textextract.NewChunkEmbedder(embedClient.Embed)
	}
	return r
}

// Definitions returns the Ollama tool schemas for all supported tools, in a
// stable order suitable for inclusion in a chat request.
func (r *Registry) Definitions() []ollama.Tool {
	defs := []ollama.Tool{
		listDirDefinition(),
		readFileDefinition(),
		readFilesDefinition(),
		readDocumentDefinition(),
		searchNameDefinition(),
		searchContentDefinition(),
		searchDocumentDefinition(),
		searchSemanticDefinition(),
		statFileDefinition(),
		getChunkDefinition(),
		findAllFilesDefinition(),
	}
	if r.visionCapable {
		defs = append(defs, readImageDefinition())
	}
	return defs
}

// Call dispatches a tool call by name with the given decoded arguments,
// returning the result to feed back to the model. search_document and search_semantic
// use ctx (for embedding HTTP calls); other tools ignore it since they're
// local filesystem operations.
func (r *Registry) Call(ctx context.Context, name string, args map[string]any) (Result, error) {
	switch name {
	case "list_dir":
		return r.listDir(args)
	case "read_file":
		return r.readFile(args)
	case "read_files":
		return r.readFiles(args)
	case "read_document":
		return r.readDocument(args)
	case "read_image":
		return r.readImage(args)
	case "search_name":
		return r.searchName(args)
	case "search_content":
		return r.searchContent(args)
	case "search_document":
		return r.searchDocument(ctx, args)
	case "search_semantic":
		return r.searchSemantic(ctx, args)
	case "stat_file":
		return r.statFile(args)
	case "get_chunk":
		return r.getChunk(args)
	case "find_all_files":
		return r.findAllFiles(args)
	default:
		return Result{}, unknownToolError(name)
	}
}
