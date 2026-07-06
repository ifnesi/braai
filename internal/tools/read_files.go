package tools

import (
	"fmt"
	"strings"
)

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
