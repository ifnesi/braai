package tools

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
)

// errStopWalk is a sentinel used internally to break out of filepath.WalkDir
// once a result limit has been reached; it is never surfaced to callers.
var errStopWalk = errors.New("stop walk: limit reached")

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
	filesInspected := 0

	walkErr := filepath.WalkDir(r.root.Abs(), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != r.root.Abs() && SkipDirNames[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if SkipDirNames[d.Name()] {
			return nil
		}

		info, statErr := d.Info()
		if statErr != nil || info.Size() > r.limits.MaxSearchFileBytes {
			return nil // skip large or unreadable files
		}

		filesInspected++
		found, isText, scanErr := scanFileForMatches(path, needle, caseSensitive, r.limits.MaxSearchResults-len(matches))
		if scanErr != nil || !isText {
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
		return Result{}, walkErr
	}

	truncated := len(matches) >= r.limits.MaxSearchResults
	out, jsonErr := json.Marshal(struct {
		Matches        []contentMatch `json:"matches"`
		FilesInspected int            `json:"files_inspected"`
		Truncated      bool           `json:"truncated,omitempty"`
	}{Matches: matches, FilesInspected: filesInspected, Truncated: truncated})
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
