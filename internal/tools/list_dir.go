package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

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
					"extensions": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Optional list of file extensions to include, e.g. [\".md\", \".txt\"]. Directories are always listed regardless of this filter so navigation still works. Case-insensitive.",
					},
					"sort_by": map[string]any{
						"type":        "string",
						"enum":        []string{"name", "modified_time"},
						"description": "Sort order for results: \"name\" (default, alphabetical) or \"modified_time\" (most recently modified first — useful for finding recent meeting notes).",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

type dirEntryInfo struct {
	Path       string `json:"path"` // relative to working dir root
	Type       string `json:"type"` // "file", "dir", "symlink", "other"
	Size       int64  `json:"size,omitempty"`
	Depth      int    `json:"depth"`
	ModifiedAt string `json:"modified_at,omitempty"`
	modTime    int64  // unexported: used for sorting only, not serialized
}

func (r *Registry) listDir(args map[string]any) (Result, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}
	depth := optionalIntArg(args, "depth", 1)
	if depth < 1 {
		depth = 1
	}
	extensions := stringSliceArg(args, "extensions")
	sortBy := optionalStringArg(args, "sort_by", "name")

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return Result{}, err
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return Result{}, fmt.Errorf("stat %q: %w", relPath, err)
	}
	if !info.IsDir() {
		return Result{}, fmt.Errorf("%q is not a directory", relPath)
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
		var modTime int64
		var modifiedAt string
		fi, statErr := d.Info()
		if statErr == nil {
			size = fi.Size()
			modTime = fi.ModTime().Unix()
			modifiedAt = fi.ModTime().Format("2006-01-02T15:04:05Z07:00")
			switch {
			case fi.IsDir():
				entryType = "dir"
			case fi.Mode()&os.ModeSymlink != 0:
				entryType = "symlink"
			case !fi.Mode().IsRegular():
				entryType = "other"
			}
		}
		if entryType != "dir" && !extensionMatches(d.Name(), extensions) {
			return nil
		}
		entries = append(entries, dirEntryInfo{
			Path:       r.root.RelPath(p),
			Type:       entryType,
			Size:       size,
			Depth:      curDepth,
			ModifiedAt: modifiedAt,
			modTime:    modTime,
		})
		return nil
	})
	if err != nil {
		return Result{}, err
	}

	switch sortBy {
	case "modified_time":
		sort.SliceStable(entries, func(i, j int) bool { return entries[i].modTime > entries[j].modTime })
	default:
		sort.SliceStable(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	}

	out, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return Result{}, err
	}
	return textResult(string(out)), nil
}

// extensionMatches reports whether name matches one of the given extensions
// (case-insensitive). An empty extensions list matches everything.
func extensionMatches(name string, extensions []string) bool {
	if len(extensions) == 0 {
		return true
	}
	ext := strings.ToLower(filepath.Ext(name))
	for _, want := range extensions {
		if strings.ToLower(want) == ext {
			return true
		}
	}
	return false
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
