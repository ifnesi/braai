package tools

import (
	"encoding/json"
	"fmt"
	"path/filepath"

	"braai/internal/ollama"
	"braai/internal/textextract"
)

// docChunkTokens is the fixed chunk size for the manifest/chunk workflow
// (read_document -> get_chunk / search_document). Keeping it constant across
// all three tools guarantees chunk indices line up regardless of caller args.
const docChunkTokens = 2000

// extractChunks extracts, cleans, normalizes and chunks a document at absPath
// using the fixed docChunkTokens size. Returns (nil, nil) when the document
// has no extractable text.
func (r *Registry) extractChunks(absPath string) ([]textextract.Chunk, error) {
	text, err := textextract.ExtractText(absPath)
	if err != nil {
		return nil, fmt.Errorf("extract text: %w", err)
	}
	text = textextract.CleanForLLM(text)
	text = textextract.NormalizeWhitespace(text)
	if text == "" {
		return nil, nil
	}
	return textextract.ChunkText(text, docChunkTokens, filepath.Base(absPath)), nil
}

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
	Text     string                      `json:"text,omitempty"`     // full text if small
	Tokens   int                         `json:"tokens,omitempty"`   // token count of the text
	Manifest []textextract.ManifestEntry `json:"manifest,omitempty"` // TOC if document too large
	Cleaned  bool                        `json:"cleaned"`
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

	// Extract text for the direct-return decision. clean/raw only affects the
	// direct-return text; the chunk workflow always uses cleaned text (via
	// extractChunks) so chunk indices are stable across tools.
	text, err := textextract.ExtractText(absPath)
	if err != nil {
		return Result{}, fmt.Errorf("extract text: %w", err)
	}
	if clean {
		text = textextract.CleanForLLM(text)
	}
	text = textextract.NormalizeWhitespace(text)
	if text == "" {
		text = "(no extractable text)"
	}

	tokens := textextract.EstimateTokens(text)

	// If text fits in the token budget, return it directly.
	if tokens <= maxTokens {
		out := readDocumentResult{
			Text:    text,
			Tokens:  tokens,
			Cleaned: clean,
		}
		jsonOut, _ := json.Marshal(out)
		return textResult(string(jsonOut)), nil
	}

	// Too large: chunk with the shared helper and cache for get_chunk.
	chunks, err := r.extractChunks(absPath)
	if err != nil {
		return Result{}, err
	}
	r.documentChunkCache[relPath] = chunks

	manifest := textextract.BuildManifest(chunks)
	out := readDocumentResult{
		Manifest: manifest,
		Cleaned:  clean,
	}
	jsonOut, _ := json.Marshal(out)
	return textResult(string(jsonOut)), nil
}
