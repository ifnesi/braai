package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

func (r *Registry) searchName(args map[string]any) (Result, error) {
	pattern, err := stringArg(args, "pattern")
	if err != nil {
		return Result{}, err
	}
	caseSensitive := optionalBoolArg(args, "case_sensitive", false)
	extensions := stringSliceArg(args, "extensions")

	needle := pattern
	if !caseSensitive {
		needle = strings.ToLower(needle)
	}

	var matches []string
	err = filepath.WalkDir(r.root.Abs(), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries rather than failing the whole search
		}
		if path == r.root.Abs() {
			return nil
		}
		if d.IsDir() && skipDirNames[d.Name()] {
			return filepath.SkipDir
		}
		if !d.IsDir() && !extensionMatches(d.Name(), extensions) {
			return nil
		}
		name := d.Name()
		if !caseSensitive {
			name = strings.ToLower(name)
		}
		if strings.Contains(name, needle) {
			matches = append(matches, r.root.RelPath(path))
			if len(matches) >= r.limits.MaxNameResults {
				return errStopWalk
			}
		}
		return nil
	})
	if err != nil && err != errStopWalk {
		return Result{}, err
	}

	truncated := len(matches) >= r.limits.MaxNameResults
	out, jsonErr := json.Marshal(struct {
		Matches   []string `json:"matches"`
		Truncated bool     `json:"truncated,omitempty"`
	}{Matches: matches, Truncated: truncated})
	if jsonErr != nil {
		return Result{}, jsonErr
	}
	return textResult(string(out)), nil
}
