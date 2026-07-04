package textextract

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
)

// ChunkEmbedder embeds document chunks and performs semantic search.
// It caches embeddings by file path+mtime+chunk count for efficiency.
type ChunkEmbedder struct {
	embedFn EmbedFn
	mu      sync.RWMutex
	cache   map[string]*CachedChunks
}

// EmbedFn is the signature of an embedding function (from Ollama).
// It takes a model name and a list of texts, returns embedding vectors.
// Each vector should be normalized (magnitude 1) by the embedFn itself.
type EmbedFn func(ctx context.Context, model string, texts []string) ([][]float32, error)

// CachedChunks holds embedded chunks for a document.
type CachedChunks struct {
	Path            string
	ModTime         int64
	ChunkCount      int
	ChunksWithEmbed []ChunkWithEmbedding
}

// NewChunkEmbedder creates a new embedder with the given embedding function.
func NewChunkEmbedder(embedFn EmbedFn) *ChunkEmbedder {
	return &ChunkEmbedder{
		embedFn: embedFn,
		cache:   make(map[string]*CachedChunks),
	}
}

// EmbedChunks embeds all chunks in a document, caching by path+mtime+count.
// If the file hasn't changed and the chunk count matches, returns the cached embedding.
func (ce *ChunkEmbedder) EmbedChunks(ctx context.Context, model string, absPath string, chunks []Chunk) ([]ChunkWithEmbedding, error) {
	// Get file metadata for cache invalidation.
	info, err := os.Stat(absPath)
	if err != nil {
		return nil, fmt.Errorf("stat file for embedding: %w", err)
	}

	cacheKey := fmt.Sprintf("%s_%d_%d", absPath, info.ModTime().Unix(), len(chunks))

	// Check cache.
	ce.mu.RLock()
	if cached, ok := ce.cache[cacheKey]; ok {
		ce.mu.RUnlock()
		return cached.ChunksWithEmbed, nil
	}
	ce.mu.RUnlock()

	// Extract texts from chunks.
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}

	// Call the embedding function. It should return normalized vectors.
	embeddings, err := ce.embedFn(ctx, model, texts)
	if err != nil {
		return nil, fmt.Errorf("embed chunks: %w", err)
	}

	if len(embeddings) != len(chunks) {
		return nil, fmt.Errorf("embedding function returned %d vectors for %d chunks", len(embeddings), len(chunks))
	}

	// Pair chunks with embeddings.
	result := make([]ChunkWithEmbedding, len(chunks))
	for i := range chunks {
		result[i] = ChunkWithEmbedding{
			Chunk:     &chunks[i],
			Embedding: embeddings[i],
		}
	}

	// Cache the result.
	ce.mu.Lock()
	ce.cache[cacheKey] = &CachedChunks{
		Path:            absPath,
		ModTime:         info.ModTime().Unix(),
		ChunkCount:      len(chunks),
		ChunksWithEmbed: result,
	}
	ce.mu.Unlock()

	return result, nil
}

// SearchChunks finds the top-K chunks most similar to the query by semantic similarity.
// The similarity threshold filters results (0-1, default 0 means no minimum).
func (ce *ChunkEmbedder) SearchChunks(ctx context.Context, model string, query string, chunksWithEmbed []ChunkWithEmbedding, topK int, threshold float32) ([]RankedChunk, error) {
	if len(chunksWithEmbed) == 0 {
		return []RankedChunk{}, nil
	}

	// Embed the query.
	queryEmbed, err := ce.embedFn(ctx, model, []string{query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	if len(queryEmbed) != 1 {
		return nil, fmt.Errorf("query embedding failed: expected 1 vector, got %d", len(queryEmbed))
	}

	queryVec := queryEmbed[0]

	// Compute similarity (cosine) between query and each chunk.
	// Since vectors are normalized, cosine similarity = dot product.
	type scored struct {
		chunk      *ChunkWithEmbedding
		similarity float32
	}
	var scores []scored

	for i := range chunksWithEmbed {
		sim := dotProduct(queryVec, chunksWithEmbed[i].Embedding)
		if sim >= threshold {
			scores = append(scores, scored{
				chunk:      &chunksWithEmbed[i],
				similarity: sim,
			})
		}
	}

	// Sort by similarity (descending).
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].similarity > scores[j].similarity
	})

	// Limit to topK.
	if len(scores) > topK {
		scores = scores[:topK]
	}

	// Build result.
	result := make([]RankedChunk, len(scores))
	for i, s := range scores {
		result[i] = RankedChunk{
			Chunk:      s.chunk.Chunk,
			Similarity: s.similarity,
			Summary:    chunkSummary(s.chunk.Chunk.Text, 120),
		}
	}

	return result, nil
}

// dotProduct computes the dot product of two vectors.
// Both vectors must be the same length. For normalized vectors, this equals cosine similarity.
func dotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(math.Max(-1, math.Min(1, sum))) // clamp to [-1, 1]
}
