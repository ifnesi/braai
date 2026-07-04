package textextract

import "strings"

// Chunk is a slice of extracted text sized for LLM consumption, with metadata
// that lets a downstream model localize and cite the content.
type Chunk struct {
	Index   int    `json:"index"`             // 1-indexed position in document
	Total   int    `json:"total"`             // total chunks in this document
	Source  string `json:"source"`            // filename
	Section string `json:"section,omitempty"` // sheet/slide name, if applicable
	Tokens  int    `json:"tokens"`            // estimated token count
	Text    string `json:"text"`              // the actual content
}

// ManifestEntry is a compact, model-facing description of one chunk. Send the
// full manifest to an LLM as a table of contents so it can decide which chunks
// to request, or use it as a human-readable overview of a document's structure.
type ManifestEntry struct {
	Index   int    `json:"index"`
	Total   int    `json:"total"`
	Source  string `json:"source"`
	Section string `json:"section,omitempty"`
	Tokens  int    `json:"tokens"`
	Summary string `json:"summary"`
}

// ChunkWithEmbedding pairs a chunk with its normalized embedding vector
// for semantic search.
type ChunkWithEmbedding struct {
	Chunk     *Chunk    `json:"chunk"`
	Embedding []float32 `json:"embedding,omitempty"` // normalized (magnitude 1)
}

// RankedChunk represents a chunk and its relevance score for search results.
type RankedChunk struct {
	Chunk      *Chunk  `json:"chunk"`
	Similarity float32 `json:"similarity"` // 0-1, higher is more relevant
	Summary    string  `json:"summary"`
}

// EstimateTokens approximates LLM token count without a tokenizer dependency.
// The ~4-chars-per-token heuristic is a good rule of thumb for English/code.
func EstimateTokens(s string) int {
	n := len([]rune(s)) / 4
	if n < 1 && strings.TrimSpace(s) != "" {
		return 1
	}
	return n
}

// NormalizeWhitespace collapses consecutive blank lines, trims each line,
// and removes trailing whitespace.
func NormalizeWhitespace(s string) string {
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	blank := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			if !blank {
				out = append(out, "")
				blank = true
			}
			continue
		}
		out = append(out, line)
		blank = false
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}
