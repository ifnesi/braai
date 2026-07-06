package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"braai/internal/cache"
)

// maxTotalChunks bounds how many chunks (across all files) one search_semantic
// call will score, to cap latency/memory on very large trees.
const maxTotalChunks = 5000

type semanticMatch struct {
	Path       string  `json:"path"`
	ChunkIndex int     `json:"chunk_index"`
	Section    string  `json:"section,omitempty"`
	Score      float64 `json:"score"`
	Excerpt    string  `json:"excerpt"`
}

func (r *Registry) searchSemantic(ctx context.Context, args map[string]any) (Result, error) {
	if r.embedClient == nil || r.chunkEmbedder == nil {
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
	// Default threshold 0: return all candidates ranked (weak matches included).
	threshold := 0.0
	if th, ok := args["threshold"]; ok {
		if f, ok := coerceFloat(th); ok {
			threshold = f
		}
	}
	extensions := stringSliceArg(args, "extensions")

	// Embed the query once per call.
	queryVecs, err := r.embedClient.Embed(ctx, r.embedModel, []string{query})
	if err != nil {
		return Result{}, fmt.Errorf("embedding request failed: %w", err)
	}
	if len(queryVecs) != 1 {
		return Result{}, fmt.Errorf("embedding request returned %d vectors for the query", len(queryVecs))
	}
	queryVec := normalize(queryVecs[0])

	paths, filesTruncated, err := r.collectSemanticFiles(extensions)
	if err != nil {
		return Result{}, err
	}
	if len(paths) == 0 {
		return textResult(`{"matches":[],"note":"no eligible files found"}`), nil
	}

	var matches []semanticMatch
	total := 0
	chunksTruncated := false

	for _, absPath := range paths {
		info, statErr := os.Stat(absPath)
		if statErr != nil {
			continue
		}
		relPath := r.root.RelPath(absPath)
		mtimeNS := info.ModTime().UnixNano()
		size := info.Size()

		metas, err := r.semanticChunkMetas(ctx, relPath, absPath, mtimeNS, size)
		if err != nil {
			return Result{}, err
		}
		if len(metas) == 0 {
			continue
		}
		if total+len(metas) > maxTotalChunks {
			chunksTruncated = true
			break
		}
		total += len(metas)

		for i := range metas {
			score := dot(queryVec, normalize(metas[i].Embedding))
			if score < threshold {
				continue
			}
			matches = append(matches, semanticMatch{
				Path:       relPath,
				ChunkIndex: metas[i].Index,
				Section:    metas[i].Section,
				Score:      score,
				Excerpt:    metas[i].Excerpt,
			})
		}
	}

	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if len(matches) > topK {
		matches = matches[:topK]
	}
	if matches == nil {
		matches = []semanticMatch{}
	}

	// Persist any newly computed embeddings/blobs.
	if r.semanticCache != nil {
		_ = r.semanticCache.Flush()
	}

	out, jsonErr := json.Marshal(struct {
		Matches   []semanticMatch `json:"matches"`
		Truncated bool            `json:"truncated,omitempty"`
	}{Matches: matches, Truncated: filesTruncated || chunksTruncated})
	if jsonErr != nil {
		return Result{}, jsonErr
	}
	return textResult(string(out)), nil
}

// semanticChunkMetas returns the chunk metadata (with normalized embeddings) for
// one file, preferring the persistent cache, then falling back to extract+embed
// (which itself uses chunkEmbedder's in-memory cache) and persisting the result.
func (r *Registry) semanticChunkMetas(ctx context.Context, relPath, absPath string, mtimeNS, size int64) ([]cache.ChunkMeta, error) {
	if r.semanticCache != nil {
		if entry, ok := r.semanticCache.Get(relPath, mtimeNS, size); ok {
			return entry.Chunks, nil
		}
	}

	chunks, err := r.extractChunks(absPath)
	if err != nil {
		// Unreadable/unsupported file: skip it rather than failing the search.
		return nil, nil
	}
	if len(chunks) == 0 {
		return nil, nil
	}

	embedded, err := r.chunkEmbedder.EmbedChunks(ctx, r.embedModel, absPath, chunks)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}

	metas := make([]cache.ChunkMeta, len(embedded))
	texts := make([]string, len(embedded))
	for i := range embedded {
		ch := embedded[i].Chunk
		metas[i] = cache.ChunkMeta{
			Index:     ch.Index,
			Section:   ch.Section,
			Tokens:    ch.Tokens,
			Excerpt:   truncateExcerpt(ch.Text, 200),
			Embedding: normalize(embedded[i].Embedding),
		}
		texts[i] = ch.Text
	}

	if r.semanticCache != nil {
		entry := &cache.FileEntry{
			RelPath:   relPath,
			ModTimeNS: mtimeNS,
			Size:      size,
			Chunks:    metas,
		}
		_ = r.semanticCache.Put(entry, texts)
	}
	return metas, nil
}

// collectSemanticFiles walks the root collecting up to MaxSemanticFiles eligible
// files (same skip/extension/size rules as the other searches). It does not read
// file bodies — extraction happens per file in semanticChunkMetas, so PDF/DOCX/
// etc. are supported, not just plaintext.
func (r *Registry) collectSemanticFiles(extensions []string) ([]string, bool, error) {
	var paths []string
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
		paths = append(paths, path)
		if len(paths) >= r.limits.MaxSemanticFiles {
			truncated = true
			return errStopWalk
		}
		return nil
	})
	if walkErr != nil && walkErr != errStopWalk {
		return nil, false, walkErr
	}
	return paths, truncated, nil
}

// normalize returns v scaled to unit length, so cosine similarity against
// another normalized vector reduces to a plain dot product.
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

// dot returns the dot product of two equal-length vectors (0 on mismatch). For
// unit vectors this equals cosine similarity.
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
