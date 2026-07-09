package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
)

// AutoContextResult is what AutoContext returns to the caller (main.go's
// turn loop, not the model): Text is ready to inject as a "tool"-role
// message, or empty if nothing qualified this turn. Keys are the dedup keys
// ("path#chunk_index") of the chunks that were actually injected, so the
// caller can add them to its exclude set for future turns in the same
// conversation.
type AutoContextResult struct {
	Text string
	Keys []string
}

// AutoContext embeds query (the user's latest message) and returns the
// highest-scoring chunks across the working directory, formatted for
// injection into the conversation before the model responds — the
// auto-retrieval counterpart to the model explicitly calling search with
// semantic=true. It reuses collectSemanticFiles/embedAndScoreFiles, so
// indexing is parallelized and reported via SetIndexProgress exactly like an
// explicit semantic search.
//
// exclude keys chunks (by the same "path#chunk_index" key returned in Keys)
// that were already injected earlier in this conversation, so the same
// passage isn't repeated turn after turn. Returns a zero-value result (no
// error) when: no embedding backend is configured; the working directory
// has no eligible files; every candidate was already excluded; or the best
// remaining candidate scores below minScore — deliberately not injecting
// low-relevance chunks into unrelated turns (e.g. a pasted stack trace)
// rather than guessing at intent from the message text.
func (r *Registry) AutoContext(ctx context.Context, query string, topK int, minScore float64, maxChars int, exclude map[string]bool) (AutoContextResult, error) {
	if r.embedClient == nil || r.chunkEmbedder == nil {
		return AutoContextResult{}, nil
	}

	queryVecs, err := r.embedClient.Embed(ctx, r.embedModel, []string{query})
	if err != nil {
		return AutoContextResult{}, fmt.Errorf("auto-context embedding failed: %w", err)
	}
	if len(queryVecs) != 1 {
		return AutoContextResult{}, fmt.Errorf("embedding request returned %d vectors for the query", len(queryVecs))
	}
	queryVec := normalize(queryVecs[0])

	paths, _, err := r.collectSemanticFiles(nil)
	if err != nil {
		return AutoContextResult{}, err
	}
	if len(paths) == 0 {
		return AutoContextResult{}, nil
	}

	// threshold 0: rank every candidate here; minScore is applied below, only
	// after excluding chunks already injected, so the relevance check reflects
	// what's actually left to offer, not what was available before exclusion.
	matches, _, err := r.embedAndScoreFiles(ctx, paths, queryVec, 0)
	if err != nil {
		return AutoContextResult{}, err
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })

	var picked []semanticMatch
	var keys []string
	for _, m := range matches {
		key := fmt.Sprintf("%s#%d", m.Path, m.ChunkIndex)
		if exclude[key] {
			continue
		}
		picked = append(picked, m)
		keys = append(keys, key)
		if len(picked) >= topK {
			break
		}
	}
	if len(picked) == 0 || picked[0].Score < minScore {
		return AutoContextResult{}, nil
	}

	body, err := json.Marshal(struct {
		Matches []semanticMatch `json:"matches"`
	}{Matches: picked})
	if err != nil {
		return AutoContextResult{}, err
	}

	text := string(body)
	if len(text) > maxChars {
		text = text[:maxChars] + "\n\n[auto-context truncated]"
	}

	// Deliberately the same shape searchSemantic returns (just {"matches":
	// [...]}), with no extra framing/explanation: the caller (main.go) wraps
	// this in a synthetic assistant-tool_calls + tool-result message pair so
	// it is indistinguishable, from the model's perspective, from a real
	// search_semantic call it made itself. Framing it any other way risks
	// exactly what testing found: a model can (correctly, per its own system
	// prompt telling it to distrust content it didn't fetch "via a tool
	// call") ignore an unsolicited tool-role message that doesn't fit that
	// shape, silently discarding relevant, correctly-retrieved context.
	return AutoContextResult{Text: text, Keys: keys}, nil
}
