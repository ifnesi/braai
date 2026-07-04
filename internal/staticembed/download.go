package staticembed

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// EnsureModel downloads tokenizer.json, model.safetensors and (best-effort)
// config.json for a Hugging Face repo into cacheDir on first use, and returns
// the local model directory. Subsequent runs reuse the cached files. Files are
// written 0600 inside a 0700 directory.
func EnsureModel(ctx context.Context, repo, cacheDir string) (string, error) {
	dir := filepath.Join(cacheDir, strings.ReplaceAll(repo, "/", "_"))
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	_ = os.Chmod(dir, 0o700)

	for _, name := range []string{"tokenizer.json", "model.safetensors"} {
		dst := filepath.Join(dir, name)
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		if err := download(ctx, hfURL(repo, name), dst); err != nil {
			return "", fmt.Errorf("download %s: %w", name, err)
		}
	}
	// config.json is optional (only used for the normalize flag).
	if cfg := filepath.Join(dir, "config.json"); !exists(cfg) {
		_ = download(ctx, hfURL(repo, "config.json"), cfg)
	}
	return dir, nil
}

func hfURL(repo, file string) string {
	return "https://huggingface.co/" + repo + "/resolve/main/" + file
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

func download(ctx context.Context, url, dst string) error {
	dctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(dctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d for %s", resp.StatusCode, url)
	}

	tmp := dst + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	_ = os.Chmod(tmp, 0o600)
	return os.Rename(tmp, dst)
}
