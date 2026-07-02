package tools

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"braai/internal/ollama"
)

func readFileDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "read_file",
			Description: "Read the contents of a text file within the working directory. Binary files are refused. Output may be truncated for very large files; an optional line range can be given to read a specific slice.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Path relative to the working directory root.",
					},
					"start_line": map[string]any{
						"type":        "integer",
						"description": "1-based line number to start reading from (optional).",
					},
					"end_line": map[string]any{
						"type":        "integer",
						"description": "1-based inclusive line number to stop reading at (optional).",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

func (r *Registry) readFile(args map[string]any) (string, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return "", err
	}
	startLine := optionalIntArg(args, "start_line", 0)
	endLine := optionalIntArg(args, "end_line", 0)

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return "", fmt.Errorf("stat %q: %w", relPath, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("%q is a directory, not a file", relPath)
	}

	isText, err := looksLikeText(absPath)
	if err != nil {
		return "", fmt.Errorf("read %q: %w", relPath, err)
	}
	if !isText {
		return "", fmt.Errorf("%q appears to be a binary file; refusing to read", relPath)
	}

	f, err := os.Open(absPath)
	if err != nil {
		return "", fmt.Errorf("open %q: %w", relPath, err)
	}
	defer f.Close()

	var b strings.Builder
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	lineNum := 0
	truncatedByBytes := false
	truncatedByRange := false
	written := 0

	for scanner.Scan() {
		lineNum++
		if startLine > 0 && lineNum < startLine {
			continue
		}
		if endLine > 0 && lineNum > endLine {
			truncatedByRange = false // exited normally due to range end, not a truncation
			break
		}
		line := fmt.Sprintf("%6d\t%s\n", lineNum, scanner.Text())
		if r.limits.MaxReadBytes >= 0 && written+len(line) > r.limits.MaxReadBytes {
			truncatedByBytes = true
			break
		}
		b.WriteString(line)
		written += len(line)
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("scan %q: %w", relPath, err)
	}

	result := b.String()
	if result == "" && lineNum == 0 {
		return "(empty file)", nil
	}
	if truncatedByBytes {
		result += fmt.Sprintf("\n[truncated: reached max-read-bytes limit of %d]\n", r.limits.MaxReadBytes)
	}
	if truncatedByRange && endLine > 0 {
		result += fmt.Sprintf("\n[stopped at requested end_line %d]\n", endLine)
	}
	return result, nil
}
