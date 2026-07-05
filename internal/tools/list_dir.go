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
			Description: "List files and directories within the working directory. Set depth to control recursion (depth 1 = immediate entries only, higher = deeper; use a large depth like 100 to list an entire tree). When the request targets specific file types (e.g. \"the PDF files\", \"markdown notes\"), you MUST pass the extensions filter instead of listing everything and filtering afterwards. To find entries whose name contains a word (e.g. \"files named budget\"), set name_contains and use a large depth to search recursively. Supports sort_by name|modified_time.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path relative to the working directory root. Use \".\" for the root itself.",
					},
					"depth": map[string]any{
						"type":        "integer",
						"description": "Recursion depth. 1 = only immediate entries (default). Higher values recurse deeper; use a large value (e.g. 100) to list all files under the path recursively.",
					},
					"extensions": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "File extensions to include, e.g. [\".md\", \".txt\"]. Include the leading dot; case-insensitive. ALWAYS set this when the user asks for specific file types (e.g. [\".pdf\"] for \"the PDF files\") so the tool returns only matching files instead of the whole directory. Directories are always listed regardless of this filter so navigation still works.",
					},
					"name_contains": map[string]any{
						"type":        "string",
						"description": "Only include entries whose name contains this substring (case-insensitive). Applies to both files and directories. Use with a large depth to find matching files anywhere under the path.",
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
	nameContains := optionalStringArg(args, "name_contains", "")
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
		if nameContains != "" && !strings.Contains(strings.ToLower(d.Name()), strings.ToLower(nameContains)) {
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

	out, err := json.Marshal(entries)
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
