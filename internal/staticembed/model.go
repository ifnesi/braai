// Package staticembed runs model2vec-style static embeddings entirely
// in-process: tokenize -> average the token rows -> L2-normalize. No server,
// no cgo, no external Go modules. A loaded *Model satisfies the same interface
// (Embed) that braai's tool registry expects from an embedding backend, so it
// is a drop-in replacement for the Ollama embedding client.
package staticembed

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Model is a loaded static embedding model (token table + tokenizer).
type Model struct {
	vocab     map[string]int
	unkID     int
	prefix    string // continuing-subword prefix, e.g. "##"
	maxChars  int
	lowercase bool

	rows      []float32 // flat [numRows*dim] embedding matrix
	dim       int
	numRows   int
	normalize bool
}

// Load reads tokenizer.json, model.safetensors and (optional) config.json from dir.
func Load(dir string) (*Model, error) {
	tk, err := loadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	rows, nRows, dim, err := readSafetensors(filepath.Join(dir, "model.safetensors"), "embeddings")
	if err != nil {
		return nil, fmt.Errorf("load embeddings: %w", err)
	}

	normalize := true // potion / model2vec models normalize by default
	if b, rerr := os.ReadFile(filepath.Join(dir, "config.json")); rerr == nil {
		var cfg struct {
			Normalize *bool `json:"normalize"`
		}
		if json.Unmarshal(b, &cfg) == nil && cfg.Normalize != nil {
			normalize = *cfg.Normalize
		}
	}

	unkID, ok := tk.vocab[tk.unk]
	if !ok {
		unkID = 0
	}
	return &Model{
		vocab:     tk.vocab,
		unkID:     unkID,
		prefix:    tk.prefix,
		maxChars:  tk.maxChars,
		lowercase: tk.lowercase,
		rows:      rows,
		dim:       dim,
		numRows:   nRows,
		normalize: normalize,
	}, nil
}

// Dim returns the embedding dimensionality.
func (m *Model) Dim() int { return m.dim }

// Embed satisfies the tools.embedder interface. The model argument is ignored
// (the model is already loaded); it exists only to match the interface shape.
// Returned vectors are L2-normalized so cosine similarity == dot product.
func (m *Model) Embed(_ context.Context, _ string, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = m.embedOne(t)
	}
	return out, nil
}

func (m *Model) embedOne(text string) []float32 {
	ids := m.tokenize(text)
	acc := make([]float64, m.dim)
	count := 0
	for _, id := range ids {
		if id < 0 || id >= m.numRows {
			continue
		}
		base := id * m.dim
		for d := 0; d < m.dim; d++ {
			acc[d] += float64(m.rows[base+d])
		}
		count++
	}
	res := make([]float32, m.dim)
	if count > 0 {
		for d := range acc {
			res[d] = float32(acc[d] / float64(count))
		}
	}
	if m.normalize {
		var sumSq float64
		for _, x := range res {
			sumSq += float64(x) * float64(x)
		}
		if sumSq > 0 {
			inv := float32(1.0 / math.Sqrt(sumSq))
			for d := range res {
				res[d] *= inv
			}
		}
	}
	return res
}

// tokenize does BERT-style basic tokenization + greedy WordPiece (no special tokens).
func (m *Model) tokenize(text string) []int {
	if m.lowercase {
		text = strings.ToLower(text)
	}
	var ids []int
	for _, word := range basicSplit(text) {
		runes := []rune(word)
		if m.maxChars > 0 && len(runes) > m.maxChars {
			ids = append(ids, m.unkID)
			continue
		}
		start := 0
		var pieces []int
		bad := false
		for start < len(runes) {
			end := len(runes)
			cur := -1
			for end > start {
				sub := string(runes[start:end])
				if start > 0 {
					sub = m.prefix + sub
				}
				if id, ok := m.vocab[sub]; ok {
					cur = id
					break
				}
				end--
			}
			if cur == -1 {
				bad = true
				break
			}
			pieces = append(pieces, cur)
			start = end
		}
		if bad {
			ids = append(ids, m.unkID)
		} else {
			ids = append(ids, pieces...)
		}
	}
	return ids
}

// basicSplit approximates BERT BasicTokenizer: split on whitespace and break out
// ASCII punctuation into single-character tokens.
func basicSplit(text string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			flush()
		case isPunct(r):
			flush()
			out = append(out, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

func isPunct(r rune) bool {
	return (r >= '!' && r <= '/') || (r >= ':' && r <= '@') ||
		(r >= '[' && r <= '`') || (r >= '{' && r <= '~')
}
