package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"braai/internal/ollama"
)

func searchNameDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "search_name",
			Description: "Search file and directory names under the working directory for a substring match. Case-insensitive by default.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{
						"type":        "string",
						"description": "Substring to search for in file/directory names.",
					},
					"case_sensitive": map[string]any{
						"type":        "boolean",
						"description": "If true, match case-sensitively. Default false.",
					},
					"extensions": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Optional list of file extensions to restrict matches to, e.g. [\".md\", \".txt\"]. Directories are always eligible regardless of this filter. Case-insensitive.",
					},
				},
				"required": []string{"pattern"},
			},
		},
	}
}

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
	out, jsonErr := json.MarshalIndent(struct {
		Matches   []string `json:"matches"`
		Truncated bool     `json:"truncated,omitempty"`
	}{Matches: matches, Truncated: truncated}, "", "  ")
	if jsonErr != nil {
		return Result{}, jsonErr
	}
	return textResult(string(out)), nil
}
