package tools

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"braai/internal/ollama"
)

// imageExtensions lists the file extensions read_image will accept. Kept
// narrow and explicit rather than sniffing content, since these are exactly
// the formats Ollama's vision models are documented to accept.
var imageExtensions = map[string]bool{
	".png":  true,
	".jpg":  true,
	".jpeg": true,
	".gif":  true,
	".webp": true,
}

func readImageDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name:        "read_image",
			Description: "Read an image (png, jpg, jpeg, gif, webp) within the working directory and attach it to the conversation so you can visually inspect it — e.g. to read text via OCR from a screenshot, or describe a diagram or photo. Only available when the active model supports vision.",
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

func (r *Registry) readImage(args map[string]any) (Result, error) {
	if !r.visionCapable {
		return Result{}, fmt.Errorf("the active model does not report vision support; pick a vision-capable model (e.g. llama3.2-vision, qwen2.5vl, gemma3, moondream) to read images")
	}

	relPath, err := stringArg(args, "path")
	if err != nil {
		return Result{}, err
	}

	ext := strings.ToLower(filepath.Ext(relPath))
	if !imageExtensions[ext] {
		return Result{}, fmt.Errorf("%q does not look like a supported image type (png, jpg, jpeg, gif, webp)", relPath)
	}

	absPath, err := r.root.Resolve(relPath)
	if err != nil {
		return Result{}, err
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return Result{}, fmt.Errorf("stat %q: %w", relPath, err)
	}
	if info.IsDir() {
		return Result{}, fmt.Errorf("%q is a directory, not an image file", relPath)
	}
	if info.Size() > r.limits.MaxImageBytes {
		return Result{}, fmt.Errorf("%q is %d bytes, over the %d byte limit for images", relPath, info.Size(), r.limits.MaxImageBytes)
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return Result{}, fmt.Errorf("read %q: %w", relPath, err)
	}

	encoded := base64.StdEncoding.EncodeToString(data)
	return Result{
		Text:   fmt.Sprintf("Image %q (%d bytes) attached below for visual inspection.", relPath, info.Size()),
		Images: []string{encoded},
	}, nil
}
