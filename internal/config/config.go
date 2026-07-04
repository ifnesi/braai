// Package config manages braai's persisted user settings under ~/.braai/settings.json.
package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Settings is the persisted user configuration. Command-line flags always
// take precedence over these values; they are only used as defaults/history.
type Settings struct {
	OllamaHost   string `json:"ollama_host,omitempty"`
	Model        string `json:"model,omitempty"`
	EmbedModel   string `json:"embed_model,omitempty"`
	MaxToolCalls int    `json:"max_tool_calls,omitempty"`

	// Semantic-search cache settings. CacheExtractedText and CacheEncryption
	// are pointers so that omitting them in settings.json means "on" (the
	// secure default) rather than the zero value false.
	CacheExtractedText *bool  `json:"cache_extracted_text,omitempty"`
	CacheCompression   string `json:"cache_compression,omitempty"` // "flate" (default) or "none"
	CacheEncryption    *bool  `json:"cache_encryption,omitempty"`
	CacheMaxBytes      int64  `json:"cache_max_bytes,omitempty"` // total blob budget; 0 = default 1 GiB
}

// Dir returns the ~/.braai directory, creating it if necessary.
func Dir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".braai")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

// Path returns the full path to settings.json.
func Path() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "settings.json"), nil
}

// CacheDir returns ~/.braai/cache, creating it with owner-only (0700) perms.
func CacheDir() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	cd := filepath.Join(dir, "cache")
	if err := os.MkdirAll(cd, 0o700); err != nil {
		return "", err
	}
	if err := os.Chmod(cd, 0o700); err != nil {
		return "", err
	}
	return cd, nil
}

// CacheKeyPath returns ~/.braai/cache.key (the AES key file; created 0600 on use).
func CacheKeyPath() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "cache.key"), nil
}

// ModelsDir returns ~/.braai/models, creating it 0700. Static embedding model
// files are cached here (one subdirectory per HF repo).
func ModelsDir() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	md := filepath.Join(dir, "models")
	if err := os.MkdirAll(md, 0o700); err != nil {
		return "", err
	}
	if err := os.Chmod(md, 0o700); err != nil {
		return "", err
	}
	return md, nil
}

// Load reads settings.json, returning an empty Settings if it does not exist.
func Load() (*Settings, error) {
	path, err := Path()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &Settings{}, nil
	}
	if err != nil {
		return nil, err
	}
	var s Settings
	if err := json.Unmarshal(data, &s); err != nil {
		return &Settings{}, nil // corrupt config: ignore rather than fail the CLI
	}
	return &s, nil
}

// Save writes settings.json atomically-ish (direct write; single-user local tool).
func Save(s *Settings) error {
	path, err := Path()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
