// Package config manages braai's persisted user settings under ~/.braai/braai.conf.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Settings is the persisted user configuration. Command-line flags always
// take precedence over these values; they are only used as defaults/history.
type Settings struct {
	OllamaHost         string
	Model              string
	EmbedModel         string
	MaxToolCalls       int
	HistoryLimit       int
	CacheExtractedText *bool
	CacheCompression   string
	CacheEncryption    *bool
	CacheMaxBytes      int64
}

// Dir returns the ~/.braai directory, creating it if necessary with owner-only (0700)
// permissions to prevent other local users from listing its sensitive contents.
func Dir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".braai")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	// Ensure permissions are 0700 even if dir already existed with looser perms.
	if err := os.Chmod(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}

// ConfPath returns the full path to braai.conf.
func ConfPath() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "braai.conf"), nil
}

// Path is deprecated; use ConfPath instead. Kept for compatibility.
func Path() (string, error) {
	return ConfPath()
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

// CommandsDir returns ~/.braai/commands (the global custom-command directory),
// creating it 0700. Per-project commands live in <working-dir>/.braai/commands.
func CommandsDir() (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	cd := filepath.Join(dir, "commands")
	if err := os.MkdirAll(cd, 0o700); err != nil {
		return "", err
	}
	return cd, nil
}

// Load reads braai.conf (key=value format with comments), returning empty Settings if not found.
func Load() (*Settings, error) {
	path, err := ConfPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &Settings{}, nil
	}
	if err != nil {
		return &Settings{}, nil // unreadable: ignore and use defaults
	}

	s := &Settings{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])

		switch key {
		case "ollama_host":
			s.OllamaHost = val
		case "model":
			s.Model = val
		case "embed_model":
			s.EmbedModel = val
		case "max_tool_calls":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxToolCalls = n
			}
		case "history_limit":
			if n, err := strconv.Atoi(val); err == nil {
				s.HistoryLimit = n
			}
		case "cache_extracted_text":
			b := parseBool(val)
			s.CacheExtractedText = &b
		case "cache_compression":
			s.CacheCompression = val
		case "cache_encryption":
			b := parseBool(val)
			s.CacheEncryption = &b
		case "cache_max_bytes":
			if n, err := strconv.ParseInt(val, 10, 64); err == nil {
				s.CacheMaxBytes = n
			}
		}
	}

	return s, nil
}

// Save updates known keys in braai.conf in place, preserving comments, blank
// lines, ordering, and any keys braai doesn't recognize. Missing known keys are
// appended at the end. Written atomically via tmp+rename.
func Save(s *Settings) error {
	path, err := ConfPath()
	if err != nil {
		return err
	}

	// Desired key -> value for keys braai manages (only those with a value).
	type kv struct{ k, v string }
	var desired []kv
	add := func(k, v string) { desired = append(desired, kv{k, v}) }
	if s.OllamaHost != "" {
		add("ollama_host", s.OllamaHost)
	}
	if s.Model != "" {
		add("model", s.Model)
	}
	if s.EmbedModel != "" {
		add("embed_model", s.EmbedModel)
	}
	if s.MaxToolCalls > 0 {
		add("max_tool_calls", strconv.Itoa(s.MaxToolCalls))
	}
	add("history_limit", strconv.Itoa(s.HistoryLimit))
	if s.CacheExtractedText != nil {
		add("cache_extracted_text", fmt.Sprintf("%v", *s.CacheExtractedText))
	}
	if s.CacheCompression != "" {
		add("cache_compression", s.CacheCompression)
	}
	if s.CacheEncryption != nil {
		add("cache_encryption", fmt.Sprintf("%v", *s.CacheEncryption))
	}
	add("cache_max_bytes", strconv.FormatInt(s.CacheMaxBytes, 10))

	want := make(map[string]string, len(desired))
	for _, d := range desired {
		want[d.k] = d.v
	}

	// Rewrite existing lines in place, preserving comments/blanks/unknown keys.
	var out []string
	seen := make(map[string]bool)
	if existing, rerr := os.ReadFile(path); rerr == nil {
		for _, line := range strings.Split(string(existing), "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || strings.HasPrefix(trimmed, "#") {
				out = append(out, line)
				continue
			}
			parts := strings.SplitN(trimmed, "=", 2)
			key := strings.TrimSpace(parts[0])
			if v, ok := want[key]; ok {
				out = append(out, key+"="+v)
				seen[key] = true
			} else {
				out = append(out, line) // preserve keys braai doesn't manage
			}
		}
	} else {
		out = append(out, "# braai configuration", "# key=value; lines starting with # are comments", "")
	}

	// Append any managed keys that weren't already present (stable order).
	for _, d := range desired {
		if !seen[d.k] {
			out = append(out, d.k+"="+d.v)
		}
	}

	content := strings.Join(out, "\n")
	if !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path) // atomic
}

// ApplyDefaults fills in missing optional fields with sensible defaults and
// returns whether settings were modified. If modified is true, caller should
// call Save() to persist the defaults back to the file.
func ApplyDefaults(s *Settings) bool {
	modified := false

	// CacheExtractedText: default true (cache text extracted from PDFs etc)
	if s.CacheExtractedText == nil {
		t := true
		s.CacheExtractedText = &t
		modified = true
	}

	// CacheCompression: default "flate" for compression
	if s.CacheCompression == "" {
		s.CacheCompression = "flate"
		modified = true
	}

	// CacheEncryption: default true (always encrypt at rest)
	if s.CacheEncryption == nil {
		t := true
		s.CacheEncryption = &t
		modified = true
	}

	// HistoryLimit: default 100
	if s.HistoryLimit == 0 {
		s.HistoryLimit = 100
		modified = true
	}

	// CacheMaxBytes: default 1 GiB. Set to explicit value so config file is truthful
	// (0 would be confusing; < 0 can be used to mean unbounded at runtime).
	if s.CacheMaxBytes == 0 {
		s.CacheMaxBytes = 1 << 30 // 1 GiB
		modified = true
	}

	return modified
}

// parseBool parses "true", "false", "yes", "no", "1", "0" case-insensitively.
func parseBool(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "true", "yes", "1", "on":
		return true
	default:
		return false
	}
}
