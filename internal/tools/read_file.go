package tools

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func (r *Registry) readFile(args map[string]any) (Result, error) {
	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}
	startLine := optionalIntArg(args, "start_line", 0)
	endLine := optionalIntArg(args, "end_line", 0)

	text, err := r.readFileText(relPath, startLine, endLine)
	if err != nil {
		return Result{}, err
	}
	return textResult(text), nil
}

// readFileText reads and line-numbers a single text file, honoring the
// registry's MaxReadBytes limit and an optional 1-based, inclusive line
// range. It is shared by the read_file and read_files tools.
func (r *Registry) readFileText(relPath string, startLine, endLine int) (string, error) {
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

	f, err := os.Open(absPath)
	if err != nil {
		return "", fmt.Errorf("open %q: %w", relPath, err)
	}
	defer f.Close()

	isText, err := looksLikeTextFile(f)
	if err != nil {
		return "", fmt.Errorf("read %q: %w", relPath, err)
	}
	if !isText {
		return "", fmt.Errorf("%q appears to be a binary file; refusing to read", relPath)
	}

	var b strings.Builder
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	lineNum := 0
	truncatedByBytes := false
	written := 0

	for scanner.Scan() {
		lineNum++
		if startLine > 0 && lineNum < startLine {
			continue
		}
		if endLine > 0 && lineNum > endLine {
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
	if endLine > 0 && lineNum > endLine {
		result += fmt.Sprintf("\n[stopped at requested end_line %d]\n", endLine)
	}
	return result, nil
}
