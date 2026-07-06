package tools

import (
	"encoding/json"
	"fmt"
	"path/filepath"

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

	maxTokens := optionalIntArg(args, "max_tokens", 2000)
	clean := optionalBoolArg(args, "clean", true)

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
