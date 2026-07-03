package tools

import (
	"encoding/json"
	"fmt"

	"braai/internal/ollama"
	"braai/internal/textextract"
)

func getChunkDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name: "get_chunk",
			Description: `Retrieve the full text of a specific chunk after reading or searching a document.

First call read_document(path) or search_document(path, query) to get manifest or search results, then use this tool to fetch the full text of a chunk by its index (1-indexed).`,
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path to the document, relative to the working directory root.",
					},
					"chunk_index": map[string]any{
						"type":        "integer",
						"description": "Which chunk to retrieve (1-indexed).",
					},
				},
				"required": []string{"path", "chunk_index"},
			},
		},
	}
}

type getChunkResult struct {
	Chunk textextract.Chunk `json:"chunk"`
	Text  string            `json:"text"`
}

func (r *Registry) getChunk(args map[string]any) (Result, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}

	chunkIndex, err := intArg(args, "chunk_index")
	if err != nil {
		return Result{}, err
	}
	if chunkIndex < 1 {
		return Result{}, fmt.Errorf("chunk_index must be >= 1")
	}

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return Result{}, err
	}

	chunks, ok := r.documentChunkCache[relPath]
	if !ok {
		chunks, err = r.extractChunks(absPath)
		if err != nil {
			return Result{}, err
		}
		if len(chunks) == 0 {
			return Result{}, fmt.Errorf("no extractable text")
		}
		r.documentChunkCache[relPath] = chunks
	}

	for _, chunk := range chunks {
		if chunk.Index == chunkIndex {
			out := getChunkResult{Chunk: chunk, Text: chunk.Text}
			jsonOut, _ := json.MarshalIndent(out, "", "  ")
			return textResult(string(jsonOut)), nil
		}
	}
	return Result{}, fmt.Errorf("chunk %d not found (document has %d chunks)", chunkIndex, len(chunks))
}
