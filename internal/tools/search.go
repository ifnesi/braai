package tools

import (
	"context"
	"strings"

	"braai/internal/ollama"
)

func searchDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name: "search",
			Description: "Search files in the working directory. Two modes:\n" +
				"- Exact (default, semantic=false): fast plain-text substring match. Returns file paths, line numbers, and excerpts. Use this for known words, identifiers, or exact phrases.\n" +
				"- Semantic (semantic=true): match by meaning using embeddings. Returns the most relevant passages with a chunk_index; call get_chunk(path, chunk_index) to read a passage in full. Use this for concepts/topics when you don't know the exact wording.\n" +
				"Set path to restrict a semantic search to a single document; omit it to search the whole tree.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "What to search for. Exact substring when semantic=false; natural-language description when semantic=true.",
					},
					"semantic": map[string]any{
						"type":        "boolean",
						"description": "false (default) = exact substring match; true = match by meaning (embeddings).",
					},
					"path": map[string]any{
						"type":        "string",
						"description": "Optional. With semantic=true, restrict the search to this single document (relative path). Ignored for exact search.",
					},
					"case_sensitive": map[string]any{
						"type":        "boolean",
						"description": "Exact search only: match case-sensitively. Default false.",
					},
					"extensions": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Semantic whole-tree search only: restrict to these file extensions, e.g. [\".md\", \".txt\"].",
					},
					"top_k": map[string]any{
						"type":        "integer",
						"description": "Semantic only: max passages to return.",
					},
					"threshold": map[string]any{
						"type":        "number",
						"description": "Semantic only: minimum similarity score (0-1) to include a passage.",
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

// search is the single model-facing search entry point. It dispatches to the
// existing implementations based on the semantic flag and whether a path is
// given, so their proven behavior (and tests) are unchanged:
//   - semantic=false            -> searchContent (exact substring, whole tree)
//   - semantic=true, no path    -> searchSemantic (embeddings, whole tree)
//   - semantic=true, with path  -> searchDocument (embeddings, one document)
func (r *Registry) search(ctx context.Context, args map[string]any) (Result, error) {
	semantic := optionalBoolArg(args, "semantic", false)
	if semantic {
		if p := optionalStringArg(args, "path", ""); strings.TrimSpace(p) != "" {
			return r.searchDocument(ctx, args)
		}
		return r.searchSemantic(ctx, args)
	}
	return r.searchContent(args)
}
