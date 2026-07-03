package textextract

import (
	"regexp"
	"strings"
)

// rePageNum matches standalone page-number lines like "3", "Page 3", "3 of 20", "3/20".
var rePageNum = regexp.MustCompile(`^\s*(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$`)

// CleanForLLM removes low-signal boilerplate before sending text to a model:
// repeated short lines (headers/footers), standalone page numbers, and blank rows.
// Section markers ("# ...") are always preserved.
func CleanForLLM(text string) string {
	lines := strings.Split(text, "\n")

	// Count occurrences of short, non-empty lines as boilerplate candidates.
	const maxBoilerLen = 80
	const repeatThreshold = 3
	freq := make(map[string]int)
	for _, ln := range lines {
		t := strings.TrimSpace(ln)
		if t == "" || len(t) > maxBoilerLen {
			continue
		}
		freq[t]++
	}

	out := make([]string, 0, len(lines))
	for _, ln := range lines {
		t := strings.TrimSpace(ln)
		if t == "" {
			out = append(out, "")
			continue
		}
		if strings.HasPrefix(t, "# ") { // keep section markers
			out = append(out, ln)
			continue
		}
		if rePageNum.MatchString(strings.ToLower(t)) {
			continue // drop standalone page numbers
		}
		if len(t) <= maxBoilerLen && freq[t] >= repeatThreshold {
			continue // drop repeated headers/footers
		}
		out = append(out, ln)
	}
	return NormalizeWhitespace(strings.Join(out, "\n"))
}

// ChunkText splits text into <=maxTokens chunks along natural boundaries.
//
// It is aware of the "# <name>" section markers that the extractors emit for
// xlsx sheets and pptx slides: such a line updates the current section, starts a
// new segment boundary, and is included in the text (unlike chunking systems
// that strip markers). Chunks never span two sections, so each chunk maps
// cleanly to a single sheet/slide for citation. Within a section, segments are
// paragraph- or row-delimited; a segment larger than maxTokens is hard-split.
func ChunkText(text string, maxTokens int, source string) []Chunk {
	if maxTokens <= 0 {
		maxTokens = 2000
	}

	// 1) Break text into section-tagged segments.
	type seg struct{ text, section string }
	var segs []seg
	section := ""
	curSection := ""
	var cur []string
	flushSeg := func() {
		if len(cur) == 0 {
			return
		}
		t := strings.TrimSpace(strings.Join(cur, "\n"))
		cur = cur[:0]
		if t == "" {
			return
		}
		for _, sub := range splitOversized(t, maxTokens) {
			segs = append(segs, seg{sub, curSection})
		}
	}
	for _, ln := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(ln)
		switch {
		case strings.HasPrefix(trimmed, "# "):
			flushSeg()
			section = strings.TrimSpace(strings.TrimPrefix(trimmed, "# "))
			curSection = section
		case trimmed == "":
			flushSeg()
		default:
			if len(cur) == 0 {
				curSection = section
			}
			cur = append(cur, ln)
		}
	}
	flushSeg()

	// 2) Pack segments into chunks, never crossing a section boundary.
	var chunks []Chunk
	var b strings.Builder
	bSection := ""
	bTokens := 0
	flush := func() {
		if b.Len() == 0 {
			return
		}
		txt := strings.TrimSpace(b.String())
		chunks = append(chunks, Chunk{
			Source:  source,
			Section: bSection,
			Text:    txt,
			Tokens:  EstimateTokens(txt),
		})
		b.Reset()
		bTokens = 0
	}
	for _, s := range segs {
		st := EstimateTokens(s.text)
		if b.Len() > 0 && (s.section != bSection || bTokens+st > maxTokens) {
			flush()
		}
		if b.Len() == 0 {
			bSection = s.section
		} else {
			b.WriteString("\n\n")
		}
		b.WriteString(s.text)
		bTokens += st
	}
	flush()

	// 3) Assign indices and return.
	for i := range chunks {
		chunks[i].Index = i + 1
		chunks[i].Total = len(chunks)
	}
	return chunks
}

// splitOversized breaks a single segment that exceeds maxTokens into smaller
// pieces, first by lines and, as a last resort, by rune count.
func splitOversized(s string, maxTokens int) []string {
	if EstimateTokens(s) <= maxTokens {
		return []string{s}
	}
	var out []string
	var cur strings.Builder
	curTokens := 0
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
			curTokens = 0
		}
	}
	for _, ln := range strings.Split(s, "\n") {
		lt := EstimateTokens(ln)
		if lt > maxTokens {
			flush()
			out = append(out, splitByRunes(ln, maxTokens*4)...)
			continue
		}
		if curTokens > 0 && curTokens+lt > maxTokens {
			flush()
		}
		if cur.Len() > 0 {
			cur.WriteString("\n")
		}
		cur.WriteString(ln)
		curTokens += lt
	}
	flush()
	return out
}

// splitByRunes splits a string by rune count. Used when a single line exceeds token budget.
func splitByRunes(s string, runeLimit int) []string {
	if runeLimit <= 0 {
		runeLimit = 8000
	}
	var out []string
	runes := []rune(s)
	for i := 0; i < len(runes); i += runeLimit {
		end := i + runeLimit
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
	}
	if len(out) == 0 && s != "" {
		out = []string{s}
	}
	return out
}

// BuildManifest derives a table of contents from chunks, with summaries for each.
func BuildManifest(chunks []Chunk) []ManifestEntry {
	entries := make([]ManifestEntry, 0, len(chunks))
	for _, c := range chunks {
		entries = append(entries, ManifestEntry{
			Index:   c.Index,
			Total:   c.Total,
			Source:  c.Source,
			Section: c.Section,
			Tokens:  c.Tokens,
			Summary: chunkSummary(c.Text, 120),
		})
	}
	return entries
}

// chunkSummary produces a one-line gist of a chunk: the first substantive
// line (skipping any section markers), whitespace-collapsed, trimmed to the
// first sentence when short, and truncated to maxLen runes.
func chunkSummary(text string, maxLen int) string {
	pick := ""
	for _, ln := range strings.Split(text, "\n") {
		t := strings.TrimSpace(ln)
		if t == "" || strings.HasPrefix(t, "# ") {
			continue
		}
		pick = t
		break
	}
	if pick == "" {
		pick = strings.TrimSpace(text)
	}
	pick = strings.Join(strings.Fields(pick), " ") // collapse tabs/spaces/newlines

	if i := firstSentenceEnd(pick); i > 0 && i <= maxLen {
		pick = strings.TrimSpace(pick[:i])
	}
	return truncateRunes(pick, maxLen)
}

// firstSentenceEnd returns the byte index just past the first sentence
// terminator (. ! ?) that is followed by a space, or 0 if none is found.
func firstSentenceEnd(s string) int {
	for i := 0; i < len(s)-1; i++ {
		switch s[i] {
		case '.', '!', '?':
			if s[i+1] == ' ' {
				return i + 1
			}
		}
	}
	return 0
}

// truncateRunes truncates a string to at most n runes.
func truncateRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	runes := []rune(s)
	if len(runes) > n {
		runes = runes[:n]
	}
	return string(runes)
}
