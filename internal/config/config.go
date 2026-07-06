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
	Mode               string // "dark" (default) or "light"
	EmbedModel         string
	MaxToolCalls       int
	HistoryLimit       int
	OllamaTimeout      int // seconds; 0 = default 300 (5 minutes)
	NumCtx             int
	KeepAlive          string
	CacheExtractedText *bool
	CacheCompression   string
	CacheEncryption    *bool
	CacheMaxBytes      int64

	// Tool limits (0 = use built-in default; see ApplyDefaults). Persisted so
	// users can tune them in braai.conf.
	MaxReadBytes       int
	MaxSearchFileBytes int64
	MaxSearchResults   int
	MaxNameResults     int
	MaxBatchFiles      int
	MaxImageBytes      int64
	MaxSemanticFiles   int
	MaxSemanticResults int
	MaxEmbedChars      int
	MaxDocumentBytes   int
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
	// Tighten permissions if the directory existed with looser perms, but
	// skip the syscall when already correct to avoid redundant work on every call.
	if fi, err := os.Stat(dir); err == nil && fi.Mode().Perm() != 0o700 {
		if err := os.Chmod(dir, 0o700); err != nil {
			return "", err
		}
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
	if fi, err := os.Stat(cd); err == nil && fi.Mode().Perm() != 0o700 {
		if err := os.Chmod(cd, 0o700); err != nil {
			return "", err
		}
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
	if fi, err := os.Stat(md); err == nil && fi.Mode().Perm() != 0o700 {
		if err := os.Chmod(md, 0o700); err != nil {
			return "", err
		}
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
		// Warn to stderr about the unreadable config and degrade to defaults.
		fmt.Fprintf(os.Stderr, "warning: unable to read config at %s: %v (using defaults)\n", path, err)
		return &Settings{}, nil
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
		case "mode":
			s.Mode = val
		case "embed_model":
			s.EmbedModel = val
		case "max_tool_calls":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxToolCalls = n
			}
		case "ollama_timeout":
			if n, err := strconv.Atoi(val); err == nil {
				s.OllamaTimeout = n
			}
		case "history_limit":
			if n, err := strconv.Atoi(val); err == nil {
				s.HistoryLimit = n
			}
		case "num_ctx":
			if n, err := strconv.Atoi(val); err == nil {
				s.NumCtx = n
			}
		case "keep_alive":
			s.KeepAlive = val
		case "max_read_bytes":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxReadBytes = n
			}
		case "max_search_file_bytes":
			if n, err := strconv.ParseInt(val, 10, 64); err == nil {
				s.MaxSearchFileBytes = n
			}
		case "max_search_results":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxSearchResults = n
			}
		case "max_name_results":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxNameResults = n
			}
		case "max_batch_files":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxBatchFiles = n
			}
		case "max_image_bytes":
			if n, err := strconv.ParseInt(val, 10, 64); err == nil {
				s.MaxImageBytes = n
			}
		case "max_semantic_files":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxSemanticFiles = n
			}
		case "max_semantic_results":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxSemanticResults = n
			}
		case "max_embed_chars":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxEmbedChars = n
			}
		case "max_document_bytes":
			if n, err := strconv.Atoi(val); err == nil {
				s.MaxDocumentBytes = n
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
	if s.Mode != "" {
		add("mode", s.Mode)
	}
	if s.EmbedModel != "" {
		add("embed_model", s.EmbedModel)
	}
	if s.MaxToolCalls > 0 {
		add("max_tool_calls", strconv.Itoa(s.MaxToolCalls))
	}
	if s.OllamaTimeout > 0 {
		add("ollama_timeout", strconv.Itoa(s.OllamaTimeout))
	}
	add("history_limit", strconv.Itoa(s.HistoryLimit))
	if s.NumCtx > 0 {
		add("num_ctx", strconv.Itoa(s.NumCtx))
	}
	if s.KeepAlive != "" {
		add("keep_alive", s.KeepAlive)
	}
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
	add("max_read_bytes", strconv.Itoa(s.MaxReadBytes))
	add("max_search_file_bytes", strconv.FormatInt(s.MaxSearchFileBytes, 10))
	add("max_search_results", strconv.Itoa(s.MaxSearchResults))
	add("max_name_results", strconv.Itoa(s.MaxNameResults))
	add("max_batch_files", strconv.Itoa(s.MaxBatchFiles))
	add("max_image_bytes", strconv.FormatInt(s.MaxImageBytes, 10))
	add("max_semantic_files", strconv.Itoa(s.MaxSemanticFiles))
	add("max_semantic_results", strconv.Itoa(s.MaxSemanticResults))
	add("max_embed_chars", strconv.Itoa(s.MaxEmbedChars))
	add("max_document_bytes", strconv.Itoa(s.MaxDocumentBytes))

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
		// New file: write a comprehensive template with all tunable limits and explanations.
		template := []string{
			"# braai configuration",
			"# key=value; lines starting with # are comments. Command-line flags override these.",
			"",
			"# ── Core ─────────────────────────────────────────────────────────────────────",
			"# URL of your local Ollama server (used for the chat model only).",
			"ollama_host=http://localhost:11434",
			"",
			"# Default chat model. Auto-detected from the first available model if blank.",
			"model=",
			"",
			"# Hugging Face repo of the static embedding model braai downloads and runs",
			"# in-process for semantic search (NOT an Ollama model). No server needed.",
			"embed_model=minishlab/potion-retrieval-32M",
			"",
			"# Max tool calls the model may make per request before braai aborts the turn.",
			"max_tool_calls=100",
			"",
			"# How long braai waits for one Ollama request, in seconds. Covers a full",
			"# streamed turn (thinking + tokens). 0/blank = default 300 (5 minutes).",
			"# Raise this (e.g. 1200) for long-running summary commands.",
			"ollama_timeout=300",
			"",
			"# ── Ollama runtime (blank/0 = use server defaults) ───────────────────────────",
			"# Context window in tokens. Raise (e.g. 16384) if long tool results get",
			"# truncated and the model \"forgets\" and re-fetches. 0 = Ollama default.",
			"num_ctx=0",
			"",
			"# How long Ollama keeps the model loaded between calls, e.g. 30m or -1 (forever).",
			"# Blank = Ollama default. Keeping it resident avoids reload latency per call.",
			"keep_alive=",
			"",
			"# ── Chat recall history ──────────────────────────────────────────────────────",
			"# Number of past prompts kept for up/down-arrow recall (encrypted at rest).",
			"history_limit=100",
			"",
			"# ── Semantic-search cache (encrypted at rest with ~/.braai/cache.key) ────────",
			"# Persist extracted document text to disk so get_chunk is instant. false = a",
			"# privacy-first mode: only embeddings/metadata cached, text re-extracted on demand.",
			"cache_extracted_text=true",
			"",
			"# Compression for cached text blobs: flate or none.",
			"cache_compression=flate",
			"",
			"# Encrypt cached text blobs at rest (AES-256-GCM). Strongly recommended.",
			"cache_encryption=true",
			"",
			"# Total on-disk budget for cache blobs before least-recently-used eviction.",
			"# 1073741824 = 1 GiB. Use -1 for unbounded.",
			"cache_max_bytes=1073741824",
			"",
			"# ── Tool limits (0 = use built-in default) ───────────────────────────────────",
			"# Max bytes read() returns for a single text file. -1 = unlimited.",
			"max_read_bytes=-1",
			"",
			"# Max size of a file that search/content will scan. 2097152 = 2 MiB.",
			"max_search_file_bytes=2097152",
			"",
			"# Max results returned by an exact (non-semantic) search.",
			"max_search_results=200",
			"",
			"# Max results returned when filtering entries by name (list_dir name_contains).",
			"max_name_results=500",
			"",
			"# Max number of files read() will accept in one batch (paths=[...]).",
			"max_batch_files=20",
			"",
			"# Max size of an image read_image will load. 10485760 = 10 MiB.",
			"max_image_bytes=10485760",
			"",
			"# Max number of files scanned in a whole-tree semantic search.",
			"max_semantic_files=200",
			"",
			"# Max passages returned by a semantic search.",
			"max_semantic_results=10",
			"",
			"# Max characters embedded per file/query for semantic search.",
			"max_embed_chars=8000",
			"",
			"# Max extracted text (bytes) per document when read() batches many docs, so a",
			"# multi-PDF read can't flood the model context. 131072 = 128 KiB.",
			"max_document_bytes=131072",
			"",
		}
		out = append(out, template...)
	}

	// Append any managed keys that weren't already present (stable order),
	// marking each as seen so a duplicate entry in `desired` isn't written twice.
	for _, d := range desired {
		if !seen[d.k] {
			out = append(out, d.k+"="+d.v)
			seen[d.k] = true
		}
	}

	content := strings.Join(out, "\n")
	if !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o600); err != nil {
		return err
	}
	if err := os.Chmod(tmp, 0o600); err != nil {
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

	// OllamaTimeout: default 300 seconds (5 minutes)
	if s.OllamaTimeout == 0 {
		s.OllamaTimeout = 300
		modified = true
	}

	// CacheMaxBytes: default 1 GiB. Set to explicit value so config file is truthful
	// (0 would be confusing; < 0 can be used to mean unbounded at runtime).
	if s.CacheMaxBytes == 0 {
		s.CacheMaxBytes = 1 << 30 // 1 GiB
		modified = true
	}

	// Tool limits: 0 means unset. max_read_bytes uses -1 (unlimited) as default.
	if s.MaxReadBytes == 0 {
		s.MaxReadBytes = -1
		modified = true
	}
	if s.MaxSearchFileBytes == 0 {
		s.MaxSearchFileBytes = 2 * 1024 * 1024
		modified = true
	}
	if s.MaxSearchResults == 0 {
		s.MaxSearchResults = 200
		modified = true
	}
	if s.MaxNameResults == 0 {
		s.MaxNameResults = 500
		modified = true
	}
	if s.MaxBatchFiles == 0 {
		s.MaxBatchFiles = 20
		modified = true
	}
	if s.MaxImageBytes == 0 {
		s.MaxImageBytes = 10 * 1024 * 1024
		modified = true
	}
	if s.MaxSemanticFiles == 0 {
		s.MaxSemanticFiles = 200
		modified = true
	}
	if s.MaxSemanticResults == 0 {
		s.MaxSemanticResults = 10
		modified = true
	}
	if s.MaxEmbedChars == 0 {
		s.MaxEmbedChars = 8000
		modified = true
	}
	if s.MaxDocumentBytes == 0 {
		s.MaxDocumentBytes = 131072
		modified = true
	}
	// NumCtx (0) and KeepAlive ("") intentionally left as "server default".

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

// ConfigDef describes one known configuration key: its type, whether a change
// can be applied immediately to the running session (Hot), whether it is
// read-only from /config (e.g. because a dedicated command handles it), and a
// short human-readable description shown by /config.
type ConfigDef struct {
	Key         string
	Type        string // "string" | "int" | "int64" | "bool"
	Section     string // grouping label for the /config listing
	Hot         bool   // true = can be applied immediately; false = restart required
	ReadOnly    bool   // true = show but refuse writes via /config
	Description string
}

// ConfigDefs is the ordered list of every setting braai knows about.
// Keys must match exactly the switch cases in Load() and SetField().
var ConfigDefs = []ConfigDef{
	// ── Core ──────────────────────────────────────────────────────────────────
	{Key: "ollama_host", Type: "string", Section: "Core", Hot: false,
		Description: "URL of the local Ollama server (used for the chat model only)."},
	{Key: "model", Type: "string", Section: "Core", Hot: false, ReadOnly: true,
		Description: "Default chat model. Use /model <name> to change."},
	{Key: "mode", Type: "string", Section: "Core", Hot: true,
		Description: "Colour scheme: dark (default, cyan prompt) or light (blue prompt, no dim — for light terminal backgrounds)."},
	{Key: "embed_model", Type: "string", Section: "Core", Hot: false,
		Description: "Hugging Face repo of the static embedding model run in-process for semantic search."},
	{Key: "max_tool_calls", Type: "int", Section: "Core", Hot: true,
		Description: "Max tool calls the model may make per request before braai aborts the turn."},
	{Key: "ollama_timeout", Type: "int", Section: "Core", Hot: false,
		Description: "HTTP timeout for Ollama requests in seconds (0 = default 300 / 5 min)."},

	// ── Ollama runtime ────────────────────────────────────────────────────────
	{Key: "num_ctx", Type: "int", Section: "Ollama runtime", Hot: true,
		Description: "Context window in tokens. Raise (e.g. 16384) if long tool results get truncated. 0 = Ollama default."},
	{Key: "keep_alive", Type: "string", Section: "Ollama runtime", Hot: true,
		Description: "How long Ollama keeps the model loaded between calls (e.g. 30m or -1). Blank = Ollama default."},

	// ── Chat recall history ───────────────────────────────────────────────────
	{Key: "history_limit", Type: "int", Section: "Chat recall history", Hot: false,
		Description: "Number of past prompts kept for up/down-arrow recall (encrypted at rest)."},

	// ── Semantic-search cache ─────────────────────────────────────────────────
	{Key: "cache_extracted_text", Type: "bool", Section: "Semantic-search cache", Hot: false,
		Description: "Persist extracted document text to disk so get_chunk is instant. false = re-extract on demand."},
	{Key: "cache_compression", Type: "string", Section: "Semantic-search cache", Hot: false,
		Description: "Compression for cached text blobs: flate or none."},
	{Key: "cache_encryption", Type: "bool", Section: "Semantic-search cache", Hot: false,
		Description: "Encrypt cached text blobs at rest with AES-256-GCM. Strongly recommended."},
	{Key: "cache_max_bytes", Type: "int64", Section: "Semantic-search cache", Hot: false,
		Description: "Total on-disk cache budget before LRU eviction. 1073741824 = 1 GiB. -1 = unbounded."},

	// ── Tool limits ───────────────────────────────────────────────────────────
	{Key: "max_read_bytes", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max bytes read() returns for a single text file. -1 = unlimited."},
	{Key: "max_search_file_bytes", Type: "int64", Section: "Tool limits", Hot: true,
		Description: "Max size of a file that search will scan. 2097152 = 2 MiB."},
	{Key: "max_search_results", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max results returned by an exact (non-semantic) search."},
	{Key: "max_name_results", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max results returned when filtering entries by name (list_dir name_contains)."},
	{Key: "max_batch_files", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max number of files read() will accept in one batch (paths=[...])."},
	{Key: "max_image_bytes", Type: "int64", Section: "Tool limits", Hot: true,
		Description: "Max size of an image read_image will load. 10485760 = 10 MiB."},
	{Key: "max_semantic_files", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max number of files scanned in a whole-tree semantic search."},
	{Key: "max_semantic_results", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max passages returned by a semantic search."},
	{Key: "max_embed_chars", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max characters embedded per file/query for semantic search."},
	{Key: "max_document_bytes", Type: "int", Section: "Tool limits", Hot: true,
		Description: "Max extracted text (bytes) per document in a batch read. 131072 = 128 KiB."},
}

// GetCurrentValue returns the current value of key from s as a printable
// string. Returns "" for unknown keys.
func GetCurrentValue(s *Settings, key string) string {
	switch key {
	case "ollama_host":
		return s.OllamaHost
	case "model":
		return s.Model
	case "mode":
		if s.Mode == "" {
			return "dark"
		}
		return s.Mode
	case "embed_model":
		return s.EmbedModel
	case "max_tool_calls":
		return strconv.Itoa(s.MaxToolCalls)
	case "ollama_timeout":
		return strconv.Itoa(s.OllamaTimeout)
	case "num_ctx":
		return strconv.Itoa(s.NumCtx)
	case "keep_alive":
		return s.KeepAlive
	case "history_limit":
		return strconv.Itoa(s.HistoryLimit)
	case "cache_extracted_text":
		if s.CacheExtractedText == nil {
			return ""
		}
		return strconv.FormatBool(*s.CacheExtractedText)
	case "cache_compression":
		return s.CacheCompression
	case "cache_encryption":
		if s.CacheEncryption == nil {
			return ""
		}
		return strconv.FormatBool(*s.CacheEncryption)
	case "cache_max_bytes":
		return strconv.FormatInt(s.CacheMaxBytes, 10)
	case "max_read_bytes":
		return strconv.Itoa(s.MaxReadBytes)
	case "max_search_file_bytes":
		return strconv.FormatInt(s.MaxSearchFileBytes, 10)
	case "max_search_results":
		return strconv.Itoa(s.MaxSearchResults)
	case "max_name_results":
		return strconv.Itoa(s.MaxNameResults)
	case "max_batch_files":
		return strconv.Itoa(s.MaxBatchFiles)
	case "max_image_bytes":
		return strconv.FormatInt(s.MaxImageBytes, 10)
	case "max_semantic_files":
		return strconv.Itoa(s.MaxSemanticFiles)
	case "max_semantic_results":
		return strconv.Itoa(s.MaxSemanticResults)
	case "max_embed_chars":
		return strconv.Itoa(s.MaxEmbedChars)
	case "max_document_bytes":
		return strconv.Itoa(s.MaxDocumentBytes)
	}
	return ""
}

// SetField parses value and assigns it to the field in s identified by key.
// Returns an error for unknown keys, read-only keys, or unparseable values.
// Does not call Save — the caller is responsible for persisting.
func SetField(s *Settings, key, value string) error {
	switch key {
	case "model":
		return fmt.Errorf("model cannot be set via /config; use /model <name> instead")
	case "mode":
		if value != "dark" && value != "light" {
			return fmt.Errorf("mode: must be %q or %q, got %q", "dark", "light", value)
		}
		s.Mode = value
	case "ollama_host":
		s.OllamaHost = value
	case "embed_model":
		s.EmbedModel = value
	case "keep_alive":
		s.KeepAlive = value
	case "cache_compression":
		if value != "flate" && value != "none" {
			return fmt.Errorf("cache_compression: must be %q or %q, got %q", "flate", "none", value)
		}
		s.CacheCompression = value
	case "max_tool_calls":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_tool_calls: must be a positive integer, got %q", value)
		}
		s.MaxToolCalls = n
	case "ollama_timeout":
		n, err := strconv.Atoi(value)
		if err != nil || n < 0 {
			return fmt.Errorf("ollama_timeout: must be a non-negative integer (seconds), got %q", value)
		}
		s.OllamaTimeout = n
	case "num_ctx":
		n, err := strconv.Atoi(value)
		if err != nil || n < 0 {
			return fmt.Errorf("num_ctx: must be a non-negative integer, got %q", value)
		}
		s.NumCtx = n
	case "history_limit":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("history_limit: must be a positive integer, got %q", value)
		}
		s.HistoryLimit = n
	case "cache_extracted_text":
		b, err := strconv.ParseBool(value)
		if err != nil {
			return fmt.Errorf("cache_extracted_text: must be true or false, got %q", value)
		}
		s.CacheExtractedText = &b
	case "cache_encryption":
		b, err := strconv.ParseBool(value)
		if err != nil {
			return fmt.Errorf("cache_encryption: must be true or false, got %q", value)
		}
		s.CacheEncryption = &b
	case "cache_max_bytes":
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return fmt.Errorf("cache_max_bytes: must be an integer, got %q", value)
		}
		s.CacheMaxBytes = n
	case "max_read_bytes":
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("max_read_bytes: must be an integer (-1 = unlimited), got %q", value)
		}
		s.MaxReadBytes = n
	case "max_search_file_bytes":
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil || n < 0 {
			return fmt.Errorf("max_search_file_bytes: must be a non-negative integer, got %q", value)
		}
		s.MaxSearchFileBytes = n
	case "max_search_results":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_search_results: must be a positive integer, got %q", value)
		}
		s.MaxSearchResults = n
	case "max_name_results":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_name_results: must be a positive integer, got %q", value)
		}
		s.MaxNameResults = n
	case "max_batch_files":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_batch_files: must be a positive integer, got %q", value)
		}
		s.MaxBatchFiles = n
	case "max_image_bytes":
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil || n < 0 {
			return fmt.Errorf("max_image_bytes: must be a non-negative integer, got %q", value)
		}
		s.MaxImageBytes = n
	case "max_semantic_files":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_semantic_files: must be a positive integer, got %q", value)
		}
		s.MaxSemanticFiles = n
	case "max_semantic_results":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_semantic_results: must be a positive integer, got %q", value)
		}
		s.MaxSemanticResults = n
	case "max_embed_chars":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_embed_chars: must be a positive integer, got %q", value)
		}
		s.MaxEmbedChars = n
	case "max_document_bytes":
		n, err := strconv.Atoi(value)
		if err != nil || n <= 0 {
			return fmt.Errorf("max_document_bytes: must be a positive integer, got %q", value)
		}
		s.MaxDocumentBytes = n
	default:
		return fmt.Errorf("unknown config key %q; run /config to list all valid keys", key)
	}
	return nil
}
