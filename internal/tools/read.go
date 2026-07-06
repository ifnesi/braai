package tools

import (
	"fmt"
	"path/filepath"
	"strings"

	"braai/internal/ollama"
	"braai/internal/textextract"
)

// documentExtensions are formats that require extraction (they are not readable
// as plain text). Everything else (.txt, .md, .csv, .json, code, etc.) is read
// directly by the plain-text reader. Images are handled by read_image, not here.
var documentExtensions = map[string]bool{
	".pdf":  true,
	".doc":  true,
	".docx": true,
	".xls":  true,
	".xlsx": true,
	".ppt":  true,
	".pptx": true,
	".rtf":  true,
	".html": true,
	".htm":  true,
}

func readDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name: "read",
			Description: "Read file contents within the working directory. Works for any readable file: plain text and code are returned directly; documents (PDF, Word, Excel, PowerPoint, HTML, RTF) have their text extracted automatically.\n" +
				"- Read one file: set path. For a text file you may pass start_line/end_line to read a slice. If a single document is very large, a manifest of chunks is returned instead of the full text — then call get_chunk(path, chunk_index) to read a chunk.\n" +
				"- Read several files at once: set paths (an array) instead of path. Prefer this over many single reads when you already know which files you need.\n" +
				"(To view an image, use read_image instead.)",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path to a single file, relative to the working directory root. Use this OR paths.",
					},
					"paths": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Paths to several files to read in one call, relative to the working directory root. Use this OR path.",
					},
					"start_line": map[string]any{
						"type":        "integer",
						"description": "Single text file only: 1-based line to start from (optional).",
					},
					"end_line": map[string]any{
						"type":        "integer",
						"description": "Single text file only: 1-based inclusive line to stop at (optional).",
					},
				},
				"required": []string{},
			},
		},
	}
}

// read is the single model-facing read entry point. It dispatches to the
// existing implementations so their behavior (and tests) are unchanged:
//   - paths given            -> readBatch (text files + extracted document text)
//   - single path, document  -> readDocument (extraction; manifest for large docs)
//   - single path, otherwise -> readFile (plain text, optional line range)
func (r *Registry) read(args map[string]any) (Result, error) {
	if _, ok := args["paths"]; ok {
		return r.readBatch(args)
	}
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, fmt.Errorf("provide either %q (single file) or %q (multiple files)", "path", "paths")
	}
	if documentExtensions[strings.ToLower(filepath.Ext(relPath))] {
		return r.readDocument(args)
	}
	return r.readFile(args)
}

// readBatch reads multiple files in one call, extracting document text where
// needed. It mirrors the old read_files output format (=== path === headers)
// and honors the same batch-size limit.
func (r *Registry) readBatch(args map[string]any) (Result, error) {
	paths := stringSliceArg(args, "paths")
	if len(paths) == 0 {
		return Result{}, fmt.Errorf("missing required argument %q (non-empty array of paths)", "paths")
	}
	if len(paths) > r.limits.MaxBatchFiles {
		return Result{}, fmt.Errorf("too many paths requested (%d); read accepts at most %d per call", len(paths), r.limits.MaxBatchFiles)
	}

	var b strings.Builder
	for _, p := range paths {
		fmt.Fprintf(&b, "=== %s ===\n", p)
		text, err := r.ReadAnyText(p)
		if err != nil {
			fmt.Fprintf(&b, "error: %v\n\n", err)
			continue
		}
		b.WriteString(text)
		b.WriteString("\n\n")
	}
	return textResult(b.String()), nil
}

// ReadAnyText returns the full text of a single file: extracted+cleaned text
// for document formats, or the plain-text reader's output otherwise. Used by
// batch reads and by the REPL's @path attachment expansion.
func (r *Registry) ReadAnyText(relPath string) (string, error) {
	if documentExtensions[strings.ToLower(filepath.Ext(relPath))] {
		absPath, err := r.root.Resolve(relPath)
		if err != nil {
			return "", err
		}
		text, err := textextract.ExtractText(absPath)
		if err != nil {
			return "", fmt.Errorf("extract text: %w", err)
		}
		text = textextract.CleanForLLM(text)
		text = textextract.NormalizeWhitespace(text)
		if text == "" {
			return "(no extractable text)", nil
		}
		cap := r.limits.MaxDocumentBytes
		if cap <= 0 {
			cap = 131072
		}
		if len(text) > cap {
			text = text[:cap] + fmt.Sprintf("\n[truncated to %d bytes — call read(path=%q) alone for the full chunked manifest]\n", cap, relPath)
		}
		return text, nil
	}
	return r.readFileText(relPath, 0, 0)
}
