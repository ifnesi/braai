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
	queryVec := normalize(queryVecs[0])

	// Cached vectors are already normalized (see ensureEmbeddings), so
	// similarity is just a dot product rather than a full cosine similarity
	// recomputing both norms on every query.
	matches := make([]semanticMatch, 0, len(candidates))
	for _, c := range candidates {
		entry := r.embedCache[c.absPath]
		matches = append(matches, semanticMatch{
			Path:    r.root.RelPath(c.absPath),
			Score:   dot(queryVec, entry.vector),
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

		// Read once and sniff the bytes already in hand, rather than opening
		// the file a second time just to check whether it looks like text.
		data, readErr := os.ReadFile(path)
		if readErr != nil || !looksLikeTextBytes(data) {
			return nil
		}
		text := string(data)
		if len(text) > r.limits.MaxEmbedChars {
			text = text[:r.limits.MaxEmbedChars]
		}

		excerpt := truncateExcerpt(text, 200)

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

// maxEmbedBatch caps how many texts go into a single Embed HTTP request, so
// a large uncached candidate set (up to MaxSemanticFiles) doesn't produce
// one enormous request body that a server or proxy might reject or truncate.
const maxEmbedBatch = 32

// ensureEmbeddings computes and caches embeddings for any candidates whose
// file has changed (or was never embedded) since the last call, in batches
// of maxEmbedBatch. Unchanged files reuse their cached vector, so repeated
// searches within one braai run only pay the embedding cost once per file.
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

	for start := 0; start < len(toEmbed); start += maxEmbedBatch {
		end := start + maxEmbedBatch
		if end > len(toEmbed) {
			end = len(toEmbed)
		}
		batch := toEmbed[start:end]

		texts := make([]string, len(batch))
		for i, c := range batch {
			texts[i] = c.text
		}

		vectors, err := r.embedClient.Embed(ctx, r.embedModel, texts)
		if err != nil {
			return err
		}

		for i, c := range batch {
			r.embedCache[c.absPath] = embedCacheEntry{modTime: modTimes[start+i], vector: normalize(vectors[i])}
		}
	}
	return nil
}

// normalize returns v scaled to unit length, so that a cosine similarity
// against another normalized vector reduces to a plain dot product — dot
// is computed once per cached vector here, rather than recomputing both
// vectors' norms on every single query.
func normalize(v []float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	if sumSq == 0 {
		return v
	}
	norm := float32(math.Sqrt(sumSq))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// dot returns the dot product of two equal-length vectors, or 0 if they're
// empty/mismatched (rather than panicking on a malformed embedding
// response). When both inputs are unit vectors (see normalize), this is
// exactly their cosine similarity.
func dot(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}
