package tools

import (
	"encoding/json"
	"fmt"
	"path/filepath"

	"braai/internal/ollama"
	"braai/internal/textextract"
)

func readDocumentDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name: "read_document",
			Description: `Extract and optionally chunk text from a document (PDF, Word, Excel, PowerPoint, HTML, CSV, JSON, RTF, plaintext, etc.) within the working directory.

If the extracted text is <= max_tokens (default 2000), returns it directly. If larger, returns a manifest/table-of-contents with summaries of each chunk, which you can then fetch individually with get_chunk(chunk_index).

By default, text is cleaned for LLM consumption (headers/footers/page numbers removed). Pass clean=false for raw text.`,
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path to the document, relative to the working directory root.",
					},
					"max_tokens": map[string]any{
						"type":        "integer",
						"description": "Approximate token budget per chunk (default 2000). If the document exceeds this, returns a manifest instead of the full text.",
						"default":     2000,
					},
					"clean": map[string]any{
						"type":        "boolean",
						"description": "Clean text by removing headers/footers/page numbers (default true). Set to false for raw extraction.",
						"default":     true,
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

type readDocumentResult struct {
	Text     string                     `json:"text,omitempty"`      // full text if small
	Tokens   int                        `json:"tokens,omitempty"`    // token count of the text
	Manifest []textextract.ManifestEntry `json:"manifest,omitempty"`  // TOC if document too large
	Cleaned  bool                       `json:"cleaned"`
}

func (r *Registry) readDocument(args map[string]any) (Result, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return Result{}, err
	}

	maxTokens := 2000
	if mt, ok := args["max_tokens"]; ok {
		if mtInt, ok := mt.(float64); ok {
			maxTokens = int(mtInt)
		}
	}

	clean := true
	if c, ok := args["clean"]; ok {
		if cBool, ok := c.(bool); ok {
			clean = cBool
		}
	}

	// Extract text.
	text, err := textextract.ExtractText(absPath)
	if err != nil {
		return Result{}, fmt.Errorf("extract text: %w", err)
	}

	// Apply cleaning if requested.
	if clean {
		text = textextract.CleanForLLM(text)
	}

	text = textextract.NormalizeWhitespace(text)
	if text == "" {
		text = "(no extractable text)"
	}

	tokens := textextract.EstimateTokens(text)
	source := filepath.Base(absPath)

	// If text fits in the token budget, return it directly.
	if tokens <= maxTokens {
		out := readDocumentResult{
			Text:    text,
			Tokens:  tokens,
			Cleaned: clean,
		}
		jsonOut, _ := json.MarshalIndent(out, "", "  ")
		return textResult(string(jsonOut)), nil
	}

	// Document is too large; chunk it and return a manifest.
	chunks := textextract.ChunkText(text, maxTokens, source)
	manifest := textextract.BuildManifest(chunks)

	// Cache the chunks for subsequent get_chunk calls.
	// We use a simple in-memory cache keyed by the relative path.
	r.documentChunkCache[relPath] = chunks

	out := readDocumentResult{
		Manifest: manifest,
		Cleaned:  clean,
	}
	jsonOut, _ := json.MarshalIndent(out, "", "  ")
	return textResult(string(jsonOut)), nil
}
