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
				},
				"required": []string{"pattern"},
			},
		},
	}
}

func (r *Registry) searchName(args map[string]any) (string, error) {
	pattern, err := stringArg(args, "pattern")
	if err != nil {
		return "", err
	}
	caseSensitive := optionalBoolArg(args, "case_sensitive", false)

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
		return "", err
	}

	truncated := len(matches) >= r.limits.MaxNameResults
	out, jsonErr := json.MarshalIndent(struct {
		Matches   []string `json:"matches"`
		Truncated bool     `json:"truncated,omitempty"`
	}{Matches: matches, Truncated: truncated}, "", "  ")
	if jsonErr != nil {
		return "", jsonErr
	}
	return string(out), nil
}
