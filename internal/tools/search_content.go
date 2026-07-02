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

func (r *Registry) searchContent(args map[string]any) (string, error) {
	query, err := stringArg(args, "query")
	if err != nil {
		return "", err
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

		isText, textErr := looksLikeText(path)
		if textErr != nil || !isText {
			return nil
		}

		filesScanned++
		found, scanErr := scanFileForMatches(path, needle, caseSensitive, r.limits.MaxSearchResults-len(matches))
		if scanErr != nil {
			return nil
		}
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
		return "", walkErr
	}

	truncated := len(matches) >= r.limits.MaxSearchResults
	out, jsonErr := json.MarshalIndent(struct {
		Matches      []contentMatch `json:"matches"`
		FilesScanned int            `json:"files_scanned"`
		Truncated    bool           `json:"truncated,omitempty"`
	}{Matches: matches, FilesScanned: filesScanned, Truncated: truncated}, "", "  ")
	if jsonErr != nil {
		return "", jsonErr
	}
	return string(out), nil
}

// scanFileForMatches reads path line by line and returns up to maxResults matches.
func scanFileForMatches(path, needle string, caseSensitive bool, maxResults int) ([]contentMatch, error) {
	if maxResults <= 0 {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

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
			results = append(results, contentMatch{Line: lineNum, Excerpt: strings.TrimSpace(truncateExcerpt(line, 200))})
			if len(results) >= maxResults {
				break
			}
		}
	}
	return results, scanner.Err()
}

func truncateExcerpt(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
