package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"braai/internal/textextract"
)

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
