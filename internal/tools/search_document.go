package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"braai/internal/ollama"
	"braai/internal/textextract"
)

func searchDocumentDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name: "search_document",
			Description: `Semantically search within a document to find the most relevant chunks.

First call read_document(path) to extract the document and get a manifest if needed. Then use this tool to search for specific topics by natural-language query. Returns the top matching chunks ranked by relevance.

Example: search_document("manual.pdf", "authentication setup") returns the 3-5 most relevant chunks about authentication.`,
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path to the document, relative to the working directory root.",
					},
					"query": map[string]any{
						"type":        "string",
						"description": "Natural-language description of what you're looking for (e.g., 'security requirements', 'how to configure LDAP').",
					},
					"top_k": map[string]any{
						"type":        "integer",
						"description": "Number of top chunks to return (default 5, max 10).",
						"default":     5,
					},
					"threshold": map[string]any{
						"type":        "number",
						"description": "Minimum similarity score (0-1) to include a chunk in results (default 0.3, meaning weak matches are excluded).",
						"default":     0.3,
					},
				},
				"required": []string{"path", "query"},
			},
		},
	}
}

type searchDocumentResult struct {
	Query   string                    `json:"query"`
	Results []textextract.RankedChunk `json:"results"`
	Count   int                       `json:"count"`
}

func (r *Registry) searchDocument(ctx context.Context, args map[string]any) (Result, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}

	query, err := stringArg(args, "query")
	if err != nil {
		return Result{}, err
	}

	topK := optionalIntArg(args, "top_k", 5)
	if topK > 10 {
		topK = 10
	}
	if topK < 1 {
		topK = 1
	}

	threshold := float32(0.3)
	if th, ok := args["threshold"]; ok {
		if thFloat, ok := coerceFloat(th); ok {
			threshold = float32(thFloat)
		}
	}

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return Result{}, err
	}

	if r.chunkEmbedder == nil {
		return Result{}, fmt.Errorf("embedding not available; configure an Ollama model with embedding support")
	}

	// Reuse cached chunks (from read_document/get_chunk) when present so
	// indices are consistent and we don't re-extract needlessly.
	chunks, ok := r.documentChunkCache[relPath]
	if !ok {
		chunks, err = r.extractChunks(absPath)
		if err != nil {
			return Result{}, err
		}
		r.documentChunkCache[relPath] = chunks
	}

	if len(chunks) == 0 {
		out := searchDocumentResult{
			Query:   query,
			Results: []textextract.RankedChunk{},
			Count:   0,
		}
		jsonOut, _ := json.Marshal(out)
		return textResult(string(jsonOut)), nil
	}

	chunksWithEmbed, err := r.chunkEmbedder.EmbedChunks(ctx, r.embedModel, absPath, chunks)
	if err != nil {
		return Result{}, fmt.Errorf("embed chunks: %w", err)
	}

	ranked, err := r.chunkEmbedder.SearchChunks(ctx, r.embedModel, query, chunksWithEmbed, topK, threshold)
	if err != nil {
		return Result{}, fmt.Errorf("search chunks: %w", err)
	}

	out := searchDocumentResult{
		Query:   query,
		Results: ranked,
		Count:   len(ranked),
	}
	jsonOut, _ := json.Marshal(out)
	return textResult(string(jsonOut)), nil
}
