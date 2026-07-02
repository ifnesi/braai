package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"braai/internal/ollama"
)

func searchSemanticDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "search_semantic",
			Description: "Search files under the working directory by meaning rather than exact text, using embeddings (e.g. \"find notes about the pricing decision\" without needing the exact words used). Ranks whole files by similarity to the query. Slower than search_content and requires the Ollama server to support embeddings; prefer search_content for exact substrings.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Natural-language description of what you're looking for.",
					},
					"top_k": map[string]any{
						"type":        "integer",
						"description": "Maximum number of ranked results to return. Default 10.",
					},
					"extensions": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Optional list of file extensions to restrict the search to, e.g. [\".md\", \".txt\"].",
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

type semanticMatch struct {
	Path    string  `json:"path"`
	Score   float64 `json:"score"`
	Excerpt string  `json:"excerpt"`
}

func (r *Registry) searchSemantic(ctx context.Context, args map[string]any) (Result, error) {
	if r.embedClient == nil {
		return Result{}, fmt.Errorf("search_semantic is unavailable: no embedding client configured")
	}

	query, err := stringArg(args, "query")
	if err != nil {
		return Result{}, err
	}
	topK := optionalIntArg(args, "top_k", r.limits.MaxSemanticResults)
	if topK <= 0 || topK > r.limits.MaxSemanticResults {
		topK = r.limits.MaxSemanticResults
	}
	extensions := stringSliceArg(args, "extensions")

	candidates, truncated, err := r.collectSemanticCandidates(extensions)
	if err != nil {
		return Result{}, err
	}
	if len(candidates) == 0 {
		return textResult(`{"matches":[],"note":"no eligible text files found"}`), nil
	}

	if err := r.ensureEmbeddings(ctx, candidates); err != nil {
		return Result{}, fmt.Errorf("embedding request failed: %w", err)
	}

	queryVecs, err := r.embedClient.Embed(ctx, r.embedModel, []string{query})
	if err != nil {
		return Result{}, fmt.Errorf("embedding request failed: %w", err)
	}
	queryVec := queryVecs[0]

	matches := make([]semanticMatch, 0, len(candidates))
	for _, c := range candidates {
		entry := r.embedCache[c.absPath]
		matches = append(matches, semanticMatch{
			Path:    r.root.RelPath(c.absPath),
			Score:   cosineSimilarity(queryVec, entry.vector),
			Excerpt: c.excerpt,
		})
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if len(matches) > topK {
		matches = matches[:topK]
	}

	out, jsonErr := json.MarshalIndent(struct {
		Matches   []semanticMatch `json:"matches"`
		Truncated bool            `json:"truncated,omitempty"`
	}{Matches: matches, Truncated: truncated}, "", "  ")
	if jsonErr != nil {
		return Result{}, jsonErr
	}
	return textResult(string(out)), nil
}

type semanticCandidate struct {
	absPath string
	text    string
	excerpt string
}

// collectSemanticCandidates walks the root collecting up to MaxSemanticFiles
// eligible text files (same skip rules as search_content), reading each
// one's content (truncated to MaxEmbedChars) for embedding.
func (r *Registry) collectSemanticCandidates(extensions []string) ([]semanticCandidate, bool, error) {
	var candidates []semanticCandidate
	truncated := false

	walkErr := filepath.WalkDir(r.root.Abs(), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != r.root.Abs() && skipDirNames[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if skipDirNames[d.Name()] || !extensionMatches(d.Name(), extensions) {
			return nil
		}

		info, statErr := d.Info()
		if statErr != nil || info.Size() == 0 || info.Size() > r.limits.MaxSearchFileBytes {
			return nil
		}

		isText, textErr := looksLikeText(path)
		if textErr != nil || !isText {
			return nil
		}

		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return nil
		}
		text := string(data)
		if len(text) > r.limits.MaxEmbedChars {
			text = text[:r.limits.MaxEmbedChars]
		}

		excerpt := text
		if len(excerpt) > 200 {
			excerpt = excerpt[:200]
		}

		candidates = append(candidates, semanticCandidate{absPath: path, text: text, excerpt: excerpt})
		if len(candidates) >= r.limits.MaxSemanticFiles {
			truncated = true
			return errStopWalk
		}
		return nil
	})
	if walkErr != nil && walkErr != errStopWalk {
		return nil, false, walkErr
	}
	return candidates, truncated, nil
}

// ensureEmbeddings computes and caches embeddings for any candidates whose
// file has changed (or was never embedded) since the last call, batching all
// of them into as few Embed calls as possible. Unchanged files reuse their
// cached vector, so repeated searches within one braai run only pay the
// embedding cost once per file.
func (r *Registry) ensureEmbeddings(ctx context.Context, candidates []semanticCandidate) error {
	var toEmbed []semanticCandidate
	var modTimes []int64

	for _, c := range candidates {
		info, err := os.Stat(c.absPath)
		if err != nil {
			continue
		}
		mtime := info.ModTime().UnixNano()
		if cached, ok := r.embedCache[c.absPath]; ok && cached.modTime == mtime {
			continue
		}
		toEmbed = append(toEmbed, c)
		modTimes = append(modTimes, mtime)
	}
	if len(toEmbed) == 0 {
		return nil
	}

	texts := make([]string, len(toEmbed))
	for i, c := range toEmbed {
		texts[i] = c.text
	}

	vectors, err := r.embedClient.Embed(ctx, r.embedModel, texts)
	if err != nil {
		return err
	}

	for i, c := range toEmbed {
		r.embedCache[c.absPath] = embedCacheEntry{modTime: modTimes[i], vector: vectors[i]}
	}
	return nil
}

// cosineSimilarity returns the cosine similarity of two equal-length
// vectors, or 0 if they're empty/mismatched (rather than panicking on a
// malformed embedding response).
func cosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
