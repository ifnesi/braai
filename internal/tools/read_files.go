package tools

import (
	"fmt"
	"strings"

	"braai/internal/ollama"
)

func readFilesDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "read_files",
			Description: "Read several text files within the working directory in one call, e.g. to summarize a batch of meeting notes or transcripts. Each file is subject to the same binary-refusal and truncation rules as read_file. Prefer this over multiple read_file calls when you already know which files you need.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"paths": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Paths relative to the working directory root.",
					},
				},
				"required": []string{"paths"},
			},
		},
	}
}

func (r *Registry) readFiles(args map[string]any) (Result, error) {
	paths := stringSliceArg(args, "paths")
	if len(paths) == 0 {
		return Result{}, fmt.Errorf("missing required argument %q (non-empty array of paths)", "paths")
	}
	if len(paths) > r.limits.MaxBatchFiles {
		return Result{}, fmt.Errorf("too many paths requested (%d); read_files accepts at most %d per call", len(paths), r.limits.MaxBatchFiles)
	}

	var b strings.Builder
	for _, p := range paths {
		fmt.Fprintf(&b, "=== %s ===\n", p)
		text, err := r.readFileText(p, 0, 0)
		if err != nil {
			fmt.Fprintf(&b, "error: %v\n\n", err)
			continue
		}
		b.WriteString(text)
		b.WriteString("\n\n")
	}
	return textResult(b.String()), nil
}
