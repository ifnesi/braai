package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"braai/internal/ollama"
)

func statFileDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "stat_file",
			Description: "Return metadata about a file or directory within the working directory: type, size, modification time, permissions, and extension.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path relative to the working directory root.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

type statResult struct {
	Path        string `json:"path"`
	Type        string `json:"type"`
	SizeBytes   int64  `json:"size_bytes"`
	ModTime     string `json:"mod_time"`
	Permissions string `json:"permissions"`
	Extension   string `json:"extension,omitempty"`
}

func (r *Registry) statFile(args map[string]any) (Result, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return Result{}, err
	}

	info, err := os.Lstat(absPath)
	if err != nil {
		return Result{}, fmt.Errorf("stat %q: %w", relPath, err)
	}

	entryType := "file"
	switch {
	case info.Mode()&os.ModeSymlink != 0:
		entryType = "symlink"
	case info.IsDir():
		entryType = "dir"
	case !info.Mode().IsRegular():
		entryType = "other"
	}

	result := statResult{
		Path:        r.root.RelPath(absPath),
		Type:        entryType,
		SizeBytes:   info.Size(),
		ModTime:     info.ModTime().Format("2006-01-02T15:04:05Z07:00"),
		Permissions: info.Mode().Perm().String(),
	}
	if entryType == "file" {
		result.Extension = filepath.Ext(absPath)
	}

	out, jsonErr := json.Marshal(result)
	if jsonErr != nil {
		return Result{}, jsonErr
	}
	return textResult(string(out)), nil
}
