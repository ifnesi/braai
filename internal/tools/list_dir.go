package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"braai/internal/ollama"
)

func listDirDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "list_dir",
			Description: "List entries (files and directories) in a directory within the working directory. Does not recurse by default.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path relative to the working directory root. Use \".\" for the root itself.",
					},
					"depth": map[string]any{
						"type":        "integer",
						"description": "Recursion depth: 1 lists only the given directory's immediate entries, 2 also lists one level of subdirectories, and so on. Default 1.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

type dirEntryInfo struct {
	Path  string `json:"path"` // relative to working dir root
	Type  string `json:"type"` // "file", "dir", "symlink", "other"
	Size  int64  `json:"size,omitempty"`
	Depth int    `json:"depth"`
}

func (r *Registry) listDir(args map[string]any) (string, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return "", err
	}
	depth := optionalIntArg(args, "depth", 1)
	if depth < 1 {
		depth = 1
	}

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return "", fmt.Errorf("stat %q: %w", relPath, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%q is not a directory", relPath)
	}

	var entries []dirEntryInfo
	err = walkLimited(absPath, 1, depth, func(p string, d os.DirEntry, curDepth int) error {
		if skipDirNames[d.Name()] {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		entryType := "file"
		var size int64
		fi, statErr := d.Info()
		if statErr == nil {
			size = fi.Size()
			switch {
			case fi.IsDir():
				entryType = "dir"
			case fi.Mode()&os.ModeSymlink != 0:
				entryType = "symlink"
			case !fi.Mode().IsRegular():
				entryType = "other"
			}
		}
		entries = append(entries, dirEntryInfo{
			Path:  r.root.RelPath(p),
			Type:  entryType,
			Size:  size,
			Depth: curDepth,
		})
		return nil
	})
	if err != nil {
		return "", err
	}

	out, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// walkLimited walks dir up to maxDepth levels (1 = immediate children only),
// invoking fn for every entry encountered (not for dir itself).
func walkLimited(dir string, curDepth, maxDepth int, fn func(path string, d os.DirEntry, depth int) error) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("read dir: %w", err)
	}
	for _, e := range entries {
		p := filepath.Join(dir, e.Name())
		if err := fn(p, e, curDepth); err != nil {
			if err == filepath.SkipDir {
				continue
			}
			return err
		}
		if e.IsDir() && curDepth < maxDepth {
			if err := walkLimited(p, curDepth+1, maxDepth, fn); err != nil {
				return err
			}
		}
	}
	return nil
}
