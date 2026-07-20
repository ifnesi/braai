package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"braai/internal/ollama"
)

func findAllFilesDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "find_all_files",
			Description: "Recursively list all files under a directory (and subdirectories). Returns a compact JSON list of all file paths. Useful for exploring directory structure and discovering files.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Directory path to search (relative to working directory root).",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

func (r *Registry) findAllFiles(args map[string]any) (Result, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return Result{}, err
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return Result{}, fmt.Errorf("stat path: %w", err)
	}

	if !info.IsDir() {
		return Result{}, fmt.Errorf("path is not a directory")
	}

	var allFiles []string
	err = filepath.WalkDir(absPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip errors, continue walking
		}

		// Skip certain directories
		if d.IsDir() && skipDirNames[d.Name()] {
			return filepath.SkipDir
		}

		// Add files (not directories)
		if !d.IsDir() {
			// Return path relative to working directory root for clarity
			relPath := r.root.RelPath(path)
			allFiles = append(allFiles, relPath)
		}

		return nil
	})

	if err != nil {
		return Result{}, fmt.Errorf("walk directory: %w", err)
	}

	// Sort for consistency
	sort.Strings(allFiles)

	// Build compact JSON response
	type response struct {
		Path  string   `json:"p"`
		Count int      `json:"c"`
		Files []string `json:"f"`
	}

	resp := response{
		Path:  relPath,
		Count: len(allFiles),
		Files: allFiles,
	}

	out, _ := marshalCompactJSON(resp)
	return textResult(out), nil
}
