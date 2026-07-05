package tools

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"braai/internal/ollama"
)

func searchContentDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "search_content",
			Description: "Search the text contents of files under the working directory for a plain-text query. Returns matching file paths, line numbers, and excerpts. Binary and very large files are skipped.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Plain text substring to search for within file contents.",
					},
					"case_sensitive": map[string]any{
						"type":        "boolean",
						"description": "If true, match case-sensitively. Default false.",
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

type contentMatch struct {
	Path    string `json:"path"`
	Line    int    `json:"line"`
	Excerpt string `json:"excerpt"`
}

func (r *Registry) searchContent(args map[string]any) (Result, error) {
	query, err := stringArg(args, "query")
	if err != nil {
		return Result{}, err
	}
	caseSensitive := optionalBoolArg(args, "case_sensitive", false)

	needle := query
	if !caseSensitive {
		needle = strings.ToLower(needle)
	}

	var matches []contentMatch
	filesScanned := 0

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
		if skipDirNames[d.Name()] {
			return nil
		}

		info, statErr := d.Info()
		if statErr != nil || info.Size() > r.limits.MaxSearchFileBytes {
			return nil // skip large or unreadable files
		}

		found, isText, scanErr := scanFileForMatches(path, needle, caseSensitive, r.limits.MaxSearchResults-len(matches))
		if scanErr != nil || !isText {
			return nil
		}
		filesScanned++
		for _, m := range found {
			m.Path = r.root.RelPath(path)
			matches = append(matches, m)
		}
		if len(matches) >= r.limits.MaxSearchResults {
			return errStopWalk
		}
		return nil
	})
	if walkErr != nil && walkErr != errStopWalk {
		return Result{}, walkErr
	}

	truncated := len(matches) >= r.limits.MaxSearchResults
	out, jsonErr := json.Marshal(struct {
		Matches      []contentMatch `json:"matches"`
		FilesScanned int            `json:"files_scanned"`
		Truncated    bool           `json:"truncated,omitempty"`
	}{Matches: matches, FilesScanned: filesScanned, Truncated: truncated})
	if jsonErr != nil {
		return Result{}, jsonErr
	}
	return textResult(string(out)), nil
}

// scanFileForMatches opens path once — sniffing it for binary content and,
// if it looks like text, scanning it line by line — returning up to
// maxResults matches. isText is false (with a nil error) for binary files,
// so the caller can skip them without treating that as a scan failure.
func scanFileForMatches(path, needle string, caseSensitive bool, maxResults int) (matches []contentMatch, isText bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer f.Close()

	isText, err = looksLikeTextFile(f)
	if err != nil || !isText {
		return nil, isText, err
	}
	if maxResults <= 0 {
		return nil, true, nil
	}

	var results []contentMatch
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		haystack := line
		if !caseSensitive {
			haystack = strings.ToLower(line)
		}
		if strings.Contains(haystack, needle) {
			results = append(results, contentMatch{Line: lineNum, Excerpt: truncateExcerpt(line, 200)})
			if len(results) >= maxResults {
				break
			}
		}
	}
	return results, true, scanner.Err()
}

// truncateExcerpt truncates s to at most max runes (not bytes), so it never
// splits a multi-byte UTF-8 rune and produce an invalid string.
func truncateExcerpt(s string, max int) string {
	s = strings.TrimSpace(s)
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max]) + "…"
}
