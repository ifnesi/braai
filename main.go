// Command braai is a read-only AI agent over a local working directory,
// using a local Ollama server for reasoning and tool selection.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/chzyer/readline"

	"braai/internal/agent"
	"braai/internal/cache"
	"braai/internal/commands"
	"braai/internal/config"
	"braai/internal/history"
	"braai/internal/ollama"
	"braai/internal/security"
	"braai/internal/staticembed"
	"braai/internal/terminal"
	"braai/internal/tools"
)

// version is the released version of braai, printed by --version.
const version = "0.4.1"

// digestPrompt is the fixed prompt submitted by --summarize / /digest.
const digestPrompt = `Walk this working directory thoroughly and produce a structured project overview.

Steps:
1. Use list_dir with a large depth (e.g. 100) to understand the full tree.
2. Read the key files: README, main entry points, configuration, docs, and any obvious "about this project" files.
3. Produce a structured overview with these sections:
   - Purpose — what this project does and who it is for
   - Structure — the directory layout and what each major part contains
   - Key files — the most important files and what each one does
   - Dependencies — external libraries or services it relies on
   - Conventions — any notable patterns, naming conventions, or architecture decisions

Be concrete and specific. Cite file paths where relevant.`

// defaultEmbedModel is the Hugging Face repo of the static embedding model braai
// downloads and runs in-process (no Ollama needed for embeddings).
const defaultEmbedModel = "minishlab/potion-retrieval-32M"

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "braai: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("braai", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		ollamaHost    = fs.String("ollama-host", "", "Ollama server base URL (default http://localhost:11434, or ~/.braai/braai.conf)")
		model         = fs.String("model", "", "Ollama model name to use (default: first model available on the server)")
		embedModel    = fs.String("embed-model", "", "Hugging Face repo of the static embedding model for semantic search. Default: minishlab/potion-retrieval-32M, or ~/.braai/braai.conf")
		workingDir    = fs.String("working-dir", "", "Root directory the agent may inspect (default: current directory)")
		prompt        = fs.String("prompt", "", "Single prompt to run non-interactively. If omitted, starts an interactive chat, unless trailing args or stdin provide a prompt.")
		verbose       = fs.Bool("verbose", false, "Print tool calls and intermediate steps")
		hideReasoning = fs.Bool("hide-reasoning", false, "Don't stream the model's reasoning/thinking trace before its answer (shown by default, on models that support it)")
		maxToolCalls  = fs.Int("max-tool-calls", 0, "Maximum number of tool calls per request (default 100, or ~/.braai/braai.conf)")
		maxReadBytes  = fs.Int("max-read-bytes", -1, "Maximum bytes read_file returns (-1 = no limit)")
		showVersion   = fs.Bool("version", false, "Print the braai version and exit")
		outputFormat  = fs.String("output", "text", `Output format: "text" (default, streamed to stdout as produced) or "json" (buffered; a single JSON object per answer with the answer, reasoning, and tool calls used)`)
		summarize     = fs.Bool("summarize", false, "Produce a structured project overview and exit (walks the tree, reads key files)")
	)
	fs.Usage = func() {
		fmt.Fprintf(stderr, `braai %s - a read-only AI agent over a local working directory

Usage:
  braai [flags] ["task or question"]
  cat question.txt | braai [flags]

Flags:
`, version)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			// -h/-help: usage was already printed by fs.Usage(); exit cleanly
			// rather than surfacing "flag: help requested" as an error.
			return nil
		}
		return err
	}

	if *showVersion {
		fmt.Fprintf(stdout, "braai %s\nCopyright (c) The braai Authors\nLicensed under the Apache License, Version 2.0\n", version)
		return nil
	}

	jsonOutput := false
	switch *outputFormat {
	case "text":
	case "json":
		jsonOutput = true
	default:
		return fmt.Errorf("invalid --output %q: must be \"text\" or \"json\"", *outputFormat)
	}

	settings, err := config.Load()
	if err != nil {
		return fmt.Errorf("load settings: %w", err)
	}

	// Apply defaults to any missing optional fields and persist if modified
	if config.ApplyDefaults(settings) {
		_ = config.Save(settings) // best-effort; don't fail if save fails
	}

	host := firstNonEmpty(*ollamaHost, settings.OllamaHost, "http://localhost:11434")
	dir := *workingDir
	if dir == "" {
		dir = "."
	}
	dir, err = expandTilde(dir)
	if err != nil {
		return fmt.Errorf("invalid working directory: %w", err)
	}

	root, err := security.NewRoot(dir)
	if err != nil {
		return fmt.Errorf("invalid working directory: %w", err)
	}

	client := ollama.NewClient(host, settings.OllamaTimeout)

	ctx := context.Background()
	selectedModel, err := resolveModel(ctx, client, *model, settings)
	if err != nil {
		return err
	}

	// Record the last-used host and effective tool-call limit as defaults for next
	// time; the model itself is persisted by chatSession.switchModel below
	// (also used at runtime by /model), so it isn't set here.
	// Precedence: explicit flag (>0) wins, then persisted settings, then the
	// built-in default. Persist the effective value without clobbering a
	// user-edited braai.conf with the flag's zero default.
	resolvedMaxToolCalls := *maxToolCalls
	if resolvedMaxToolCalls <= 0 {
		resolvedMaxToolCalls = settings.MaxToolCalls
	}
	if resolvedMaxToolCalls <= 0 {
		resolvedMaxToolCalls = agent.DefaultMaxToolCalls
	}

	settings.OllamaHost = host
	settings.MaxToolCalls = resolvedMaxToolCalls

	resolvedEmbedModel := firstNonEmpty(*embedModel, settings.EmbedModel, defaultEmbedModel)
	settings.EmbedModel = resolvedEmbedModel

	// Resolve semantic-cache options (secure defaults: cache on, compressed,
	// encrypted, 1 GiB budget). Pointer bools default to true when omitted.
	cacheText := true
	if settings.CacheExtractedText != nil {
		cacheText = *settings.CacheExtractedText
	}
	cacheEncrypt := true
	if settings.CacheEncryption != nil {
		cacheEncrypt = *settings.CacheEncryption
	}
	cacheCompression := settings.CacheCompression
	if cacheCompression == "" {
		cacheCompression = "flate"
	}
	cacheMaxBytes := settings.CacheMaxBytes
	// ApplyDefaults ensures this is never 0, but < 0 means unbounded (pass through).
	if cacheMaxBytes == 0 {
		cacheMaxBytes = 1 << 30 // 1 GiB (should not happen if ApplyDefaults ran)
	}

	// Load the in-process static embedding model (downloaded from Hugging Face
	// on first use). Non-fatal: if it fails, chat still works and semantic
	// search reports a clear "unavailable" error instead of crashing.
	var embedModelObj *staticembed.Model
	if modelsDir, mdErr := config.ModelsDir(); mdErr == nil {
		dlSpinner := terminal.NewSpinner(stderr, terminal.Detect(stderr))
		dlSpinner.SetLabel("Downloading embedding model…")
		dlSpinner.Start()
		dir, dErr := staticembed.EnsureModel(ctx, resolvedEmbedModel, modelsDir)
		dlSpinner.Stop()
		if dErr == nil {
			if m, lErr := staticembed.Load(dir); lErr == nil {
				embedModelObj = m
			} else {
				fmt.Fprintf(stderr, "warning: could not load embedding model %q: %v (semantic search disabled)\n", resolvedEmbedModel, lErr)
			}
		} else {
			fmt.Fprintf(stderr, "warning: could not fetch embedding model %q: %v (semantic search disabled)\n", resolvedEmbedModel, dErr)
		}
	}

	var semanticCache *cache.Cache
	if cacheDir, cerr := config.CacheDir(); cerr == nil {
		keyPath, _ := config.CacheKeyPath()
		c, oerr := cache.Open(cacheDir, keyPath, root.Abs(), resolvedEmbedModel, cache.Options{
			CacheText:   cacheText,
			Compression: cacheCompression,
			Encrypt:     cacheEncrypt,
			MaxBytes:    cacheMaxBytes,
		})
		if oerr == nil {
			semanticCache = c
		} else if *verbose {
			fmt.Fprintf(stderr, "warning: semantic cache disabled: %v\n", oerr)
		}
	}

	// Persist the cache index (embeddings + LRU bookkeeping) once on exit, not
	// only after a search_semantic call.
	if semanticCache != nil {
		defer func() { _ = semanticCache.Flush() }()
	}

	// Build tool limits from persisted settings (defaulted by ApplyDefaults).
	limits := tools.Limits{
		MaxReadBytes:       settings.MaxReadBytes,
		MaxSearchFileBytes: settings.MaxSearchFileBytes,
		MaxSearchResults:   settings.MaxSearchResults,
		MaxNameResults:     settings.MaxNameResults,
		MaxBatchFiles:      settings.MaxBatchFiles,
		MaxImageBytes:      settings.MaxImageBytes,
		MaxSemanticFiles:   settings.MaxSemanticFiles,
		MaxSemanticResults: settings.MaxSemanticResults,
		MaxEmbedChars:      settings.MaxEmbedChars,
		MaxDocumentBytes:   settings.MaxDocumentBytes,
	}
	// CLI flag still wins for read bytes when explicitly provided.
	if *maxReadBytes != -1 {
		limits.MaxReadBytes = *maxReadBytes
	}

	// In JSON mode, output is buffered and printed once as a single object
	// rather than streamed, so the agent's streaming writer is discarded.
	agentStdout := stdout
	colorLevel := terminal.Detect(stdout)
	if jsonOutput {
		agentStdout = io.Discard
		colorLevel = terminal.None
	}

	session := newChatSession(chatSessionOptions{
		client:        client,
		root:          root,
		limits:        limits,
		settings:      settings,
		embedModel:    resolvedEmbedModel,
		embedder:      embedModelObj,
		maxToolCalls:  resolvedMaxToolCalls,
		verbose:       *verbose,
		showReasoning: !*hideReasoning,
		workingDir:    dir,
		stdout:        agentStdout,
		colorLevel:    colorLevel,
		verboseWriter: stderr,
		semanticCache: semanticCache,
	})
	if err := session.switchModel(ctx, selectedModel); err != nil {
		return err
	}

	trailing := strings.TrimSpace(strings.Join(fs.Args(), " "))
	initialPrompt := firstNonEmpty(*prompt, trailing)

	if *summarize {
		if initialPrompt != "" {
			return fmt.Errorf("--summarize cannot be combined with --prompt or a trailing argument")
		}
		initialPrompt = digestPrompt
	}

	if initialPrompt == "" && !terminal.IsTerminal(stdin) {
		// stdin is piped and no explicit prompt/trailing args given: treat all of stdin as the prompt.
		data, readErr := io.ReadAll(stdin)
		if readErr != nil {
			return fmt.Errorf("read stdin: %w", readErr)
		}
		initialPrompt = strings.TrimSpace(string(data))
	}

	if initialPrompt != "" {
		return runOnce(ctx, session, initialPrompt, stdout, jsonOutput)
	}

	return runChat(ctx, session, jsonOutput)
}

// chatSessionOptions holds all construction-time options for newChatSession,
// replacing the previous 14-positional-arg signature to prevent silent
// mis-ordering of same-typed arguments.
type chatSessionOptions struct {
	client        *ollama.Client
	root          *security.Root
	limits        tools.Limits
	settings      *config.Settings
	embedModel    string
	embedder      *staticembed.Model
	maxToolCalls  int
	verbose       bool
	showReasoning bool
	workingDir    string
	stdout        io.Writer
	colorLevel    terminal.Level
	verboseWriter io.Writer
	semanticCache *cache.Cache
}

// chatSession bundles everything needed to (re)build an agent for a given
// model, so /model can switch models at runtime — including rechecking
// vision support and context length, which are model-specific — without
// restarting the process.
type chatSession struct {
	client        *ollama.Client
	root          *security.Root
	limits        tools.Limits
	settings      *config.Settings
	embedModel    string
	embedder      *staticembed.Model
	maxToolCalls  int
	numCtx        int
	keepAlive     string
	verbose       bool
	showReasoning bool
	stdout        io.Writer
	colorLevel    terminal.Level
	verboseWriter io.Writer
	workingDir    string

	model     string
	modelInfo ollama.ModelInfo
	ag        *agent.Agent
	registry  *tools.Registry
	cache     *cache.Cache
	hist      *history.Store
}

func newChatSession(opts chatSessionOptions) *chatSession {
	return &chatSession{
		client:        opts.client,
		root:          opts.root,
		limits:        opts.limits,
		settings:      opts.settings,
		embedModel:    opts.embedModel,
		embedder:      opts.embedder,
		maxToolCalls:  opts.maxToolCalls,
		numCtx:        opts.settings.NumCtx,
		keepAlive:     opts.settings.KeepAlive,
		verbose:       opts.verbose,
		showReasoning: opts.showReasoning,
		workingDir:    opts.workingDir,
		stdout:        opts.stdout,
		colorLevel:    opts.colorLevel,
		verboseWriter: opts.verboseWriter,
		cache:         opts.semanticCache,
	}
}

// newIndexProgressPrinter returns a tools.IndexProgressFunc that renders a
// live progress bar to w while search_semantic embeds files not yet cached
// (e.g. the first semantic search in a fresh working directory, or after
// many files changed). Shown regardless of --verbose and unconditionally —
// like the embedding-model download spinner in run(), this is a user-facing
// wait indicator, not a debug trace, and not independently configurable:
// whenever indexing actually runs, the user sees it happening.
//
// The callback is invoked by Registry.reportIndexProgress, which serializes
// calls with its own mutex even though embedding runs on a worker pool, so
// this function doesn't need any locking of its own. done==0 always marks
// the start of a call (embedAndScoreFiles reports it once, synchronously,
// before spawning workers) and done==total marks the one call that observed
// the final file complete, so both edges are reliable without extra state.
func newIndexProgressPrinter(w io.Writer, lv terminal.Level) tools.IndexProgressFunc {
	if lv == terminal.None {
		// Non-terminal stderr (redirected/piped): a \r-overwritten bar would
		// spam a log file with one line per file, so print just a start and
		// a completion note instead.
		return func(done, total int) {
			if total <= 0 {
				return
			}
			switch {
			case done == 0:
				fmt.Fprintf(w, "Indexing %d file(s) for semantic search...\n", total)
			case done >= total:
				fmt.Fprintln(w, "Indexing complete.")
			}
		}
	}

	const barWidth = 24
	return func(done, total int) {
		if total <= 0 {
			return
		}
		if done >= total {
			fmt.Fprint(w, "\r\x1b[2K") // clear the line rather than leave a stale 100% bar
			return
		}
		filled := done * barWidth / total
		bar := strings.Repeat("#", filled) + strings.Repeat("-", barWidth-filled)
		fmt.Fprintf(w, "\r\x1b[2K%s", terminal.Dim(lv, fmt.Sprintf("Indexing files: [%s] %d/%d", bar, done, total)))
	}
}

// switchModel rebuilds the tool registry and agent for model (re-checking
// vision support and context length, since both are model-specific) and
// persists it to ~/.braai/braai.conf as the new default.
func (s *chatSession) switchModel(ctx context.Context, model string) error {
	info, err := s.client.ShowModel(ctx, model)
	if err != nil && s.verbose {
		fmt.Fprintf(s.verboseWriter, "warning: could not check capabilities for %s, assuming no vision support and no context-length warnings: %v\n", model, err)
	}

	var embedClient tools.Embedder
	if s.embedder != nil {
		embedClient = s.embedder
	}
	fetchCfg := tools.FetchURLConfig{
		Enabled:        s.settings.FetchURLEnabled != nil && *s.settings.FetchURLEnabled,
		HTTPSOnly:      s.settings.FetchURLHTTPSOnly == nil || *s.settings.FetchURLHTTPSOnly,
		MaxBytes:       s.settings.FetchURLMaxBytes,
		TimeoutSeconds: s.settings.FetchURLTimeoutSeconds,
	}
	registry := tools.NewRegistry(s.root, s.limits, info.HasCapability("vision"), fetchCfg, embedClient, s.embedModel)
	registry.SetCache(s.cache)
	// Always wired up, not gated by a setting: whenever indexing actually
	// runs (explicit search or auto-context), the user sees it happening —
	// that's a transparency/fair-use property, not a preference to opt out
	// of independently of whether the work itself happens.
	registry.SetIndexProgress(newIndexProgressPrinter(s.verboseWriter, terminal.Detect(s.verboseWriter)))
	ag := agent.New(s.client, registry, agent.Options{
		Model:         model,
		MaxToolCalls:  s.maxToolCalls,
		Verbose:       s.verbose,
		VerboseWriter: s.verboseWriter,
		ShowReasoning: s.showReasoning,
		Stdout:        s.stdout,
		ColorLevel:    s.colorLevel,
		ContextLength: info.ContextLength,
		NumCtx:        s.numCtx,
		KeepAlive:     s.keepAlive,
	})

	s.model = model
	s.modelInfo = info
	s.registry = registry
	s.ag = ag

	s.settings.Model = model
	if err := config.Save(s.settings); err != nil && s.verbose {
		fmt.Fprintf(s.verboseWriter, "warning: could not save settings: %v\n", err)
	}
	return nil
}

// applyConfigField syncs live session state from s.settings after a hot-
// applicable config key has been written. It rebuilds the agent and registry
// via switchModel so that agent.Options and tools.NewRegistry both see the
// updated values — the rebuild is cheap (no model download).
func (s *chatSession) applyConfigField(ctx context.Context, key string) error {
	switch key {
	case "mode":
		// Handled inline in runChat (rl.SetPrompt); no agent rebuild needed.
		return nil
	case "max_tool_calls":
		s.maxToolCalls = s.settings.MaxToolCalls
	case "num_ctx":
		s.numCtx = s.settings.NumCtx
	case "keep_alive":
		s.keepAlive = s.settings.KeepAlive
	case "max_read_bytes",
		"max_search_file_bytes",
		"max_search_results",
		"max_name_results",
		"max_batch_files",
		"max_image_bytes",
		"max_semantic_files",
		"max_semantic_results",
		"max_embed_chars",
		"max_document_bytes":
		s.limits = tools.Limits{
			MaxReadBytes:       s.settings.MaxReadBytes,
			MaxSearchFileBytes: s.settings.MaxSearchFileBytes,
			MaxSearchResults:   s.settings.MaxSearchResults,
			MaxNameResults:     s.settings.MaxNameResults,
			MaxBatchFiles:      s.settings.MaxBatchFiles,
			MaxImageBytes:      s.settings.MaxImageBytes,
			MaxSemanticFiles:   s.settings.MaxSemanticFiles,
			MaxSemanticResults: s.settings.MaxSemanticResults,
			MaxEmbedChars:      s.settings.MaxEmbedChars,
			MaxDocumentBytes:   s.settings.MaxDocumentBytes,
		}
	case "fetch_url_enabled",
		"fetch_url_https_only",
		"fetch_url_max_bytes",
		"fetch_url_timeout_seconds":
		// fetch_url config is read directly from s.settings in switchModel;
		// no local field update needed — just fall through to the rebuild below.
	case "auto_context_enabled",
		"auto_context_top_k",
		"auto_context_min_score",
		"auto_context_max_chars":
		// maybeInjectAutoContext reads these straight from s.settings on every
		// turn, so no agent/registry rebuild (and the Ollama round-trip that
		// entails) is needed at all — unlike the tool-limit keys above.
		return nil
	}
	return s.switchModel(ctx, s.model)
}

// resolveModel picks the model to use: explicit flag wins, otherwise the
// first model available on the Ollama server. Errors out if none are available.
func resolveModel(ctx context.Context, client *ollama.Client, flagModel string, settings *config.Settings) (string, error) {
	if flagModel != "" {
		return flagModel, nil
	}

	available, err := client.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("could not reach ollama to list models: %w", err)
	}
	if len(available) == 0 {
		return "", fmt.Errorf("no models available on the ollama server; pull one first (e.g. `ollama pull llama3.1`)")
	}

	// Prefer the previously used model if it is still installed.
	if settings.Model != "" {
		for _, m := range available {
			if m == settings.Model {
				return m, nil
			}
		}
	}
	return available[0], nil
}

// runOnce executes a single prompt. In text mode the answer is streamed
// directly to stdout by ag.Run as it arrives, so nothing further is printed;
// in JSON mode ag.Run's Stdout was set to io.Discard by the caller, so the
// buffered result is printed here instead.
func runOnce(ctx context.Context, session *chatSession, prompt string, stdout io.Writer, jsonOutput bool) error {
	history := []ollama.Message{
		agent.SystemMessage(),
		{Role: "user", Content: prompt},
	}
	// A one-shot invocation is a single turn, so there's no prior conversation
	// to dedupe against — a fresh, empty exclude set is correct here.
	history = maybeInjectAutoContext(ctx, session.registry, session.settings, history, prompt, map[string]bool{})

	start := time.Now()
	result, err := session.ag.Run(ctx, history)
	elapsed := time.Since(start)
	if err != nil {
		return err
	}
	if jsonOutput {
		return printJSONResult(stdout, result)
	}
	// Print citations + elapsed-time footer in text mode.
	fmt.Fprintln(stdout)
	if citations := extractCitations(result.ToolCalls); len(citations) > 0 {
		fmt.Fprintf(stdout, "  Sources: %s\n", strings.Join(citationLinks(citations), " "))
	}
	fmt.Fprintf(stdout, "  %s\n", terminal.Cyan(session.colorLevel, formatElapsed(elapsed)))
	return nil
}

// maybeInjectAutoContext calls registry.AutoContext for the user's latest
// message and, if it returns non-empty text, appends a synthetic pair of
// messages to history — an assistant message requesting a "search" tool
// call, followed by the "tool" role result — so the model sees relevant
// passages before responding, without having to call search itself.
//
// This mimics the exact shape a real search_semantic call would leave in
// history (assistant tool_calls, then a tool result), rather than a lone
// unsolicited "tool" message. That distinction matters: testing found that
// at least one model (gemma4:e4b) silently ignored a correctly-retrieved,
// clearly relevant chunk when it arrived as an orphan tool message — almost
// certainly because the system prompt tells every model "if you have not
// seen something via a tool call, you do not know it," so a rule-following
// model can (correctly, per that instruction) distrust content it never
// asked for. Framing it as a call the model "made" sidesteps that.
//
// injected tracks chunk keys ("path#chunk_index") already shown earlier in
// this conversation, so the same passage isn't repeated turn after turn;
// callers own that map's lifetime (reset it whenever they reset history,
// e.g. on /clear).
//
// No-ops (returns history unchanged) when auto_context_enabled is false, no
// embedding backend is configured, the working directory has no eligible
// files, or nothing scored above the relevance threshold — see
// tools.Registry.AutoContext for the exact conditions. Errors from
// AutoContext itself (e.g. an embedding request failing) are swallowed
// rather than aborting the turn: auto-context is a best-effort enhancement,
// not something that should turn a working conversation into a hard failure.
func maybeInjectAutoContext(ctx context.Context, registry *tools.Registry, settings *config.Settings, history []ollama.Message, userMsg string, injected map[string]bool) []ollama.Message {
	if settings.AutoContextEnabled == nil || !*settings.AutoContextEnabled {
		return history
	}
	result, err := registry.AutoContext(ctx, userMsg, settings.AutoContextTopK, settings.AutoContextMinScore, settings.AutoContextMaxChars, injected)
	if err != nil || result.Text == "" {
		return history
	}
	for _, k := range result.Keys {
		injected[k] = true
	}
	return append(history,
		ollama.Message{
			Role: "assistant",
			ToolCalls: []ollama.ToolCall{{Function: ollama.ToolCallFunction{
				Name:      "search",
				Arguments: map[string]any{"query": userMsg, "semantic": true},
			}}},
		},
		ollama.Message{
			Role:     "tool",
			Content:  result.Text,
			ToolName: "search",
		},
	)
}

// printJSONResult encodes an agent.RunResult as a single compact JSON
// object on one line (answer, reasoning if any, and the tool calls used).
// Users can pipe to jq for pretty-printing: | jq .
func printJSONResult(w io.Writer, result agent.RunResult) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false) // keep <, >, & readable in code/HTML answers
	return enc.Encode(result)
}

// promptForMode returns the readline prompt string for the given colour level
// and mode setting. In dark mode (or when mode is unset) the prompt is cyan;
// in light mode it uses blue, which remains legible on white/light backgrounds
// where cyan can be near-invisible. When colourLevel is None, no ANSI codes
// are emitted regardless of mode.
func promptForMode(lv terminal.Level, mode string) string {
	if lv == terminal.None {
		return ">>> "
	}
	if mode == "light" {
		return terminal.Blue(lv, ">>> ")
	}
	return terminal.Cyan(lv, ">>> ")
}

// runChat drives an interactive prompt with readline-style line editing:
// left/right arrows, Ctrl-A/Ctrl-E to jump to the start/end of the line, and
// Ctrl-C to clear the current input instead of killing the process (Ctrl + d
// or /bye/exit/quit leave the chat). Slash-commands include /clear, /copy, /cache,
// /tools, /model, /save, /forget-history, and /help. When jsonOutput is set,
// each answer is printed as a buffered JSON object instead of streamed text.
func runChat(ctx context.Context, session *chatSession, jsonOutput bool) error {
	historyLimit := session.settings.HistoryLimit
	if historyLimit <= 0 {
		historyLimit = 100
	}

	// Note: We set readline's HistoryLimit to 10x our persistent limit so that
	// readline's in-memory buffer doesn't interfere with our persistence logic.
	// We manage history persistence ourselves via history.Store, so readline's
	// limit shouldn't crop our entries during the session.
	readlineHistoryLimit := historyLimit * 10

	promptNormal := promptForMode(session.colorLevel, session.settings.Mode)
	rl, err := readline.NewEx(&readline.Config{
		Prompt:       promptNormal,
		HistoryLimit: readlineHistoryLimit,
		HistoryFile:  "", // Empty disables readline's file persistence; we manage history ourselves
		AutoComplete: &braaiCompleter{root: session.root},
	})
	if err != nil {
		return fmt.Errorf("start interactive prompt: %w", err)
	}
	defer rl.Close()

	// Load the encrypted recall history and seed readline's in-memory history so
	// up/down-arrow recall works across sessions — without ever writing a
	// plaintext history file to disk.
	keyPath, _ := config.CacheKeyPath()
	if hist, herr := history.Open(historyFilePath(), keyPath, historyLimit); herr == nil {
		session.hist = hist
		histLines := hist.Lines()
		for _, h := range histLines {
			_ = rl.SaveHistory(h)
		}
	} else if session.verbose {
		fmt.Fprintf(os.Stderr, "warning: could not open encrypted chat history: %v\n", herr)
	}

	// Banner: mascot on the left, info on the right, side-by-side.
	// The mascot is 5 lines; info rows are assigned to lines 0, 1, 2, 4.
	mascot := []string{
		` ╭◠◠◠◠◠╮`,
		` |_____|`,
		` [◕ ‿ ◕]`,
		`<╞═════╡>`,
		` |_____|`,
	}
	info := []string{
		terminal.Bold(session.colorLevel, "braai") + " " + terminal.Bold(session.colorLevel, version),
		"Working directory: " + terminal.Cyan(session.colorLevel, session.workingDir),
		"Model: " + terminal.Green(session.colorLevel, session.model),
		terminal.Dim(session.colorLevel, "Licensed under the Apache License 2.0"),
		terminal.Dim(session.colorLevel, "Use Ctrl + d or /bye to exit, or /help for commands."),
	}
	// Pad each mascot line to a fixed visual width so the info column aligns cleanly.
	const mascotWidth = 10
	for i, m := range mascot {
		pad := strings.Repeat(" ", mascotWidth-len([]rune(m)))
		fmt.Fprintf(rl.Stdout(), "%s%s  %s\n", m, pad, info[i])
	}
	fmt.Fprintln(rl.Stdout())
	printHelp(rl.Stdout(), session.colorLevel)
	fmt.Fprintln(rl.Stdout())

	history := []ollama.Message{agent.SystemMessage()}
	// injected tracks auto-context chunk keys ("path#chunk_index") already
	// shown this conversation, so the same passage isn't repeated turn after
	// turn. Reset alongside history whenever /clear starts a fresh
	// conversation — a chunk the model has "forgotten" should be eligible
	// for re-injection.
	injected := map[string]bool{}
	for {
		rl.SetPrompt(promptForMode(session.colorLevel, session.settings.Mode)) // reset to normal after any previous error
		line, err := rl.Readline()
		if errors.Is(err, readline.ErrInterrupt) {
			// Ctrl-C: reset any open ANSI styling (e.g. dim codes from thinking),
			// show exit hint, and reprompt rather than exiting.
			fmt.Fprint(rl.Stdout(), terminal.Reset(session.colorLevel))
			fmt.Fprintf(rl.Stdout(), "\nUse Ctrl + d or /bye to exit, or /help for commands.\n")
			continue
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Persist to the encrypted recall history (skip bare exit synonyms).
		if line != "exit" && line != "quit" && line != "/bye" {
			_ = session.hist.Add(line)
		}
		if line == "exit" || line == "quit" || line == "/bye" {
			return nil
		}

		if strings.HasPrefix(line, "/") {
			// /digest: expand to the fixed digest prompt and fall through to submit.
			if line == "/digest" {
				fmt.Fprintf(rl.Stdout(), "%s\n\n", terminal.Dim(session.colorLevel, digestPrompt))
				line = digestPrompt
			} else if line == "/cmd" || strings.HasPrefix(line, "/cmd ") {
				// Custom prompt-template commands: /cmd [name [args...]].
				// A named command expands to a prompt and falls through to run as a
				// normal user turn; listing/errors are handled inside and loop.
				expanded, run := session.expandCmd(rl.Stdout(), line, history)
				if !run {
					continue
				}
				fmt.Fprintf(rl.Stdout(), "%s\n\n", terminal.Dim(session.colorLevel, expanded))
				line = expanded // submit the expanded template as this turn's prompt
			} else {
				if line == "/clear" {
					injected = map[string]bool{}
				}
				history = handleSlashCommand(ctx, rl, line, history, session)
				continue
			}
		}

		// Expand @path tokens inline before submitting to the model.
		// expandAtTokens returns an error (and we abort the turn) if any
		// token cannot be resolved or read. The original unexpanded line is
		// already persisted to session.hist above, so recall history stays compact.
		expanded, expandErr := expandAtTokens(line, session, rl.Stdout())
		if expandErr != nil {
			fmt.Fprintf(rl.Stdout(), "%s\n", terminal.Red(session.colorLevel, expandErr.Error()))
			rl.SetPrompt(terminal.Red(session.colorLevel, ">>> "))
			continue
		}

		history = append(history, ollama.Message{Role: "user", Content: expanded})
		// preAutoContextLen marks the position right after the user's own
		// message, before any auto-context injection. Captured so the
		// injected pair (if any) can be stripped back out below once this
		// turn is done with it — auto-context re-retrieves fresh every turn
		// anyway, so there's nothing to gain from resending an old turn's
		// retrieved passages in every future request. injected (the dedup
		// set) is intentionally left alone: it tracks "already shown once,
		// don't repeat" independent of whether that content still physically
		// sits in history.
		preAutoContextLen := len(history)
		history = maybeInjectAutoContext(ctx, session.registry, session.settings, history, expanded, injected)
		injectedCount := len(history) - preAutoContextLen
		// In text mode the answer streams straight to stdout as it arrives;
		// in JSON mode it was buffered (Stdout was io.Discard) and is
		// printed here instead. session.ag is re-read fresh each turn since
		// /model may have swapped it out.
		sp := terminal.NewSpinner(rl.Stdout(), session.colorLevel)
		session.ag.SetSpinner(sp)
		sp.Start()

		// Create a cancellable context for this agent run so Ctrl-C can interrupt it.
		runCtx, cancel := context.WithCancel(ctx)
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT)
		turnDone := make(chan struct{})
		defer close(turnDone) // always unblock watcher goroutine, even on panic
		go func() {
			select {
			case <-sigChan:
				cancel()
			case <-turnDone:
				// Run finished normally; exit without leaking this goroutine.
			}
		}()

		start := time.Now()
		result, err := session.ag.Run(runCtx, history)
		elapsed := time.Since(start)
		signal.Stop(sigChan)
		cancel()

		if err != nil {
			sp.Stop() // erase spinner before printing error
			// The request that would have used this turn's auto-context
			// injection failed, so there's nothing to keep it for — roll
			// history back to before the injection rather than leaving it
			// stuck there permanently (history isn't reassigned from
			// result.History below on this path, so without this it would
			// otherwise persist into every future turn for no benefit).
			if injectedCount > 0 {
				history = history[:preAutoContextLen:preAutoContextLen]
			}
			if errors.Is(err, context.Canceled) {
				// Context cancelled by Ctrl-C: reset terminal styling that was open
				// during agent execution (e.g. dim codes from thinking output).
				fmt.Fprint(rl.Stdout(), terminal.Reset(session.colorLevel))
				fmt.Fprintln(rl.Stdout(), terminal.Dim(session.colorLevel, "^C  (interrupted)"))
				continue
			}
			fmt.Fprintf(rl.Stdout(), "%s\n", terminal.Red(session.colorLevel, err.Error()))
			rl.SetPrompt(terminal.Red(session.colorLevel, ">>> "))
			continue
		}
		if jsonOutput {
			if err := printJSONResult(rl.Stdout(), result); err != nil {
				fmt.Fprintf(rl.Stdout(), "%s\n", terminal.Red(session.colorLevel, "error encoding JSON output: "+err.Error()))
			}
		} else {
			// Print a dim citations footer when any file-reading tools were used,
			// then how long the turn took.
			if citations := extractCitations(result.ToolCalls); len(citations) > 0 {
				links := citationLinks(citations)
				fmt.Fprintf(rl.Stdout(), "%s\n", terminal.Dim(session.colorLevel,
					"  Sources: "+strings.Join(links, " ")))
			}
			fmt.Fprintf(rl.Stdout(), "  %s\n", terminal.Cyan(session.colorLevel, formatElapsed(elapsed)))
		}
		history = result.History
		if injectedCount > 0 {
			// Drop this turn's auto-context injection from what's carried
			// into the next turn's request: keep everything up to and
			// including the user's message, skip the injected pair, then
			// keep everything the model itself added afterward (its own
			// tool calls and final answer). The dedup set above already
			// recorded these chunks were shown, independent of this.
			history = append(history[:preAutoContextLen:preAutoContextLen], result.History[preAutoContextLen+injectedCount:]...)
		}
	}
}

// printHelp writes the slash-command listing to out. Called both on startup
// (as part of the banner) and when the user types /help.
func printHelp(out io.Writer, colorLevel terminal.Level) {
	type helpEntry struct{ cmd, desc string }
	entries := []helpEntry{
		{"/clear", "reset the conversation history"},
		{"/forget-history", "erase ~/.braai/chat_history (the up/down recall history)"},
		{"/tools [full]", "list tools available to the model (full: also show arguments)"},
		{"/tree [<glob>]", "show working directory as an ASCII tree; optional glob filters root entries (e.g. /tree internal*)"},
		{"/digest", "produce a structured project overview (walks tree, reads key files)"},
		{"/model [<name>]", "show models or switch to <name> and save as default"},
		{"/config [<key> [<value>]]", "list / show / change settings  (e.g. /config mode light)"},
		{"/save <file>", "save the conversation transcript to a Markdown file"},
		{"/export json <file>", "save conversation as a JSON array"},
		{"/copy [last]", "copy full conversation (or last answer) to clipboard"},
		{"/cache [clear]", "show semantic-search cache stats (clear: wipe it)"},
		{"/cmd [<name> [args...]]", "run a custom prompt template (/cmd to list)"},
		{"@<path>", "inline a file's content into the prompt sent to the model"},
		{"", "  supported: text, PDF, Word, Excel, HTML, and more"},
		{"", "  multiple @tokens allowed per message (e.g. @a.txt @b.pdf)"},
		{"", "  escape spaces with \\ (e.g. @my\\ file.txt); Tab-completes paths"},
		{"/help", "show this message"},
		{"/bye, exit, quit", "leave the chat (Ctrl + d also works)"},
	}
	maxW := 0
	for _, e := range entries {
		if len(e.cmd) > maxW {
			maxW = len(e.cmd)
		}
	}
	fmt.Fprintln(out, "Commands:")
	for _, e := range entries {
		pad := strings.Repeat(" ", maxW-len(e.cmd)+2)
		fmt.Fprintf(out, "  %s%s%s\n", terminal.Bold(colorLevel, e.cmd), pad, e.desc)
	}
}

// handleSlashCommand processes a chat REPL command (a line starting with
// "/") and returns the (possibly modified) history to continue the loop
// with. Unknown commands are reported but otherwise harmless.
func handleSlashCommand(ctx context.Context, rl *readline.Instance, line string, history []ollama.Message, session *chatSession) []ollama.Message {
	out := rl.Stdout()
	fields := strings.Fields(line)
	cmd := fields[0]

	switch cmd {
	case "/help":
		printHelp(out, session.colorLevel)
		return history

	case "/clear":
		if session.colorLevel != terminal.None {
			// ANSI: clear screen (2J) and move cursor to top-left (H), so the
			// reset is visible immediately rather than just resetting state
			// the user can't see any effect of. Gated on colorLevel (not just
			// "is this a terminal") so NO_COLOR also suppresses this, since
			// it's still an ANSI escape sequence.
			fmt.Fprint(out, "\x1b[2J\x1b[H")
		}
		fmt.Fprintln(out, "Conversation history cleared.")
		// Mirror the startup banner so the user knows which model is active.
		fmt.Fprintf(out, "%s %s\n%s %s\n\n",
			"Model:", terminal.Green(session.colorLevel, session.model),
			"Working directory:", terminal.Cyan(session.colorLevel, session.workingDir))
		return []ollama.Message{agent.SystemMessage()}

	case "/forget-history":
		// This is the persisted up/down recall history (encrypted at
		// ~/.braai/chat_history), distinct from /clear's conversation context —
		// wipe both the on-disk file and the in-memory copy readline is holding.
		if session.hist != nil {
			if err := session.hist.Clear(); err != nil {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not erase chat history: "+err.Error()))
				return history
			}
		}
		rl.ResetHistory()
		fmt.Fprintln(out, "Command history erased.")
		return history

	case "/tools":
		full := len(fields) >= 2 &&
			(fields[1] == "full" || fields[1] == "-v" || fields[1] == "all")
		defs := session.registry.Definitions()
		fmt.Fprintf(out, "Available tools (%d):\n", len(defs))
		for _, t := range defs {
			fmt.Fprintf(out, "\n  %s\n", terminal.Bold(session.colorLevel, t.Function.Name))
			fmt.Fprintf(out, "    %s\n", t.Function.Description)
			if full {
				printToolArgs(out, session.colorLevel, t.Function.Parameters)
			}
		}
		if !full {
			fmt.Fprintln(out, "\nTip: /tools full  also shows each tool's arguments")
		}
		return history

	case "/model":
		available, err := session.client.ListModels(ctx)
		if err != nil {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not list models from ollama: "+err.Error()))
			return history
		}

		if len(fields) == 1 {
			fmt.Fprintf(out, "Current model: %s\n", terminal.Bold(session.colorLevel, session.model))
			fmt.Fprintln(out, "Available models:")
			for _, m := range available {
				if m == session.model {
					fmt.Fprintf(out, "  %s %s\n",
						terminal.Green(session.colorLevel, "✓"),
						terminal.Bold(session.colorLevel, m))
				} else {
					fmt.Fprintf(out, "  %s %s\n",
						terminal.Dim(session.colorLevel, " "),
						terminal.Dim(session.colorLevel, m))
				}
			}
			return history
		}

		target := fields[1]
		found := false
		for _, m := range available {
			if m == target {
				found = true
				break
			}
		}
		if !found {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, fmt.Sprintf("unknown model %q", target)))
			fmt.Fprintln(out, "Available models:")
			for _, m := range available {
				fmt.Fprintf(out, "  %s\n", m)
			}
			return history
		}

		if err := session.switchModel(ctx, target); err != nil {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, fmt.Sprintf("could not switch to model %q: %v", target, err)))
			return history
		}
		fmt.Fprintf(out, "Switched to model %s (saved as default).\n", target)
		return history

	case "/save":
		if len(fields) < 2 {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "usage: /save <file>"))
			return history
		}
		if err := saveTranscript(fields[1], history); err != nil {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not save transcript: "+err.Error()))
		} else {
			fmt.Fprintf(out, "saved conversation to %s\n", fields[1])
		}
		return history

	case "/copy":
		if len(fields) >= 2 && fields[1] == "last" {
			// /copy last — copy only the most recent assistant answer.
			last := lastAssistantMessage(history)
			if last == "" {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "no answer to copy yet"))
				return history
			}
			if err := copyToClipboard(last); err != nil {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not copy to clipboard: "+err.Error()))
			} else {
				fmt.Fprintln(out, "last answer copied to clipboard")
			}
			return history
		}
		// /copy — copy the full conversation transcript.
		transcript, err := formatTranscript(history)
		if err != nil {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not format transcript: "+err.Error()))
			return history
		}
		if err := copyToClipboard(transcript); err != nil {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not copy to clipboard: "+err.Error()))
		} else {
			fmt.Fprintln(out, "conversation copied to clipboard")
		}
		return history

	case "/export":
		if len(fields) < 3 || fields[1] != "json" {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "usage: /export json <file>"))
			return history
		}
		if err := saveTranscriptJSON(fields[2], history); err != nil {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not export: "+err.Error()))
		} else {
			fmt.Fprintf(out, "exported conversation to %s\n", fields[2])
		}
		return history

	case "/tree":
		// Parse optional glob pattern: /tree [<glob>]
		// The glob may include a path prefix (e.g. "internal/ag*"), in which case
		// the directory part is navigated into and only the final name component
		// is matched. Any trailing slash is stripped before processing.
		var globPattern string
		if len(fields) >= 2 {
			globPattern = strings.TrimSuffix(fields[1], "/")
		}

		if globPattern == "" {
			// No filter: show the full tree as before.
			fmt.Fprintln(out, ".")
			printTree(out, session.workingDir, "", false)
		} else {
			// Split into directory prefix + name glob.
			// e.g. "internal/ag*" → globDir="internal", nameGlob="ag*"
			// e.g. "internal*"    → globDir="",          nameGlob="internal*"
			globDir := filepath.Dir(globPattern)
			nameGlob := filepath.Base(globPattern)
			if globDir == "." {
				globDir = ""
			}

			// Resolve through session.root rather than a raw filepath.Join, so
			// a pattern like "/tree ../../etc*" can't walk the REPL outside
			// the working directory (the same confinement every model tool
			// already gets via security.Root).
			searchRoot := session.workingDir
			if globDir != "" {
				resolved, rerr := session.root.Resolve(globDir)
				if rerr != nil {
					fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, rerr.Error()))
					return history
				}
				searchRoot = resolved
			}

			entries, err := os.ReadDir(searchRoot)
			if err != nil {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not read directory: "+err.Error()))
				return history
			}
			var matched []os.DirEntry
			for _, e := range entries {
				name := e.Name()
				if tools.SkipDirNames[name] || strings.HasPrefix(name, ".") {
					continue
				}
				ok, err := filepath.Match(nameGlob, name)
				if err != nil {
					fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, fmt.Sprintf("invalid glob pattern %q: %v", globPattern, err)))
					return history
				}
				if ok {
					matched = append(matched, e)
				}
			}
			if len(matched) == 0 {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, fmt.Sprintf("no entries matching %q", globPattern)))
				return history
			}
			// Print a synthetic root label showing the navigated path.
			displayRoot := "."
			if globDir != "" {
				displayRoot = globDir + "/"
			}
			fmt.Fprintln(out, displayRoot)
			for i, e := range matched {
				last := i == len(matched)-1
				connector := "├── "
				childPrefix := "│   "
				if last {
					connector = "└── "
					childPrefix = "    "
				}
				label := e.Name()
				if e.IsDir() {
					label += "/"
				}
				fmt.Fprintf(out, "%s%s\n", connector, label)
				if e.IsDir() {
					printTree(out, filepath.Join(searchRoot, e.Name()), childPrefix, false)
				}
			}
		}
		return history

	case "/cache":
		if session.cache == nil {
			fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "semantic cache is disabled"))
			return history
		}
		if len(fields) >= 2 && fields[1] == "clear" {
			if err := session.cache.Clear(); err != nil {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not clear cache: "+err.Error()))
			} else {
				fmt.Fprintln(out, "semantic cache cleared")
			}
			return history
		}
		files, chunks, bytes := session.cache.Status()
		fmt.Fprintf(out, "semantic cache: %d files, %d chunks, %.1f MiB (use /cache clear to wipe)\n",
			files, chunks, float64(bytes)/(1024*1024))
		return history

	case "/config":
		switch len(fields) {
		case 1:
			// /config — list all settings grouped by section.
			var currentSection string
			for _, def := range config.ConfigDefs {
				if def.Section != currentSection {
					currentSection = def.Section
					fmt.Fprintf(out, "\n%s\n", terminal.Bold(session.colorLevel, "── "+def.Section+" "))
				}
				val := config.GetCurrentValue(session.settings, def.Key)
				if val == "" {
					val = terminal.Dim(session.colorLevel, "(unset)")
				}
				readonlyMark := ""
				if def.ReadOnly {
					readonlyMark = terminal.Dim(session.colorLevel, " (read-only)")
				}
				hotMark := ""
				if !def.Hot && !def.ReadOnly {
					hotMark = terminal.Dim(session.colorLevel, " *")
				}
				fmt.Fprintf(out, "  %-30s %s%s%s\n",
					terminal.Bold(session.colorLevel, def.Key),
					terminal.Cyan(session.colorLevel, val),
					readonlyMark,
					hotMark,
				)
			}
			fmt.Fprintf(out, "\n%s\n", terminal.Dim(session.colorLevel, "* requires restart to take effect   |   /config <key> <value> to change"))

		case 2:
			// /config <key> — show a single key's value and description.
			key := fields[1]
			var found *config.ConfigDef
			for i := range config.ConfigDefs {
				if config.ConfigDefs[i].Key == key {
					found = &config.ConfigDefs[i]
					break
				}
			}
			if found == nil {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, fmt.Sprintf("unknown config key %q; run /config to list all valid keys", key)))
				return history
			}
			val := config.GetCurrentValue(session.settings, key)
			if val == "" {
				val = "(unset)"
			}
			fmt.Fprintf(out, "%s = %s\n%s\n",
				terminal.Bold(session.colorLevel, key),
				terminal.Cyan(session.colorLevel, val),
				terminal.Dim(session.colorLevel, found.Description),
			)
			if found.ReadOnly {
				fmt.Fprintf(out, "%s\n", terminal.Dim(session.colorLevel, "  (read-only — use /model <name> to change)"))
			} else if !found.Hot {
				fmt.Fprintf(out, "%s\n", terminal.Dim(session.colorLevel, "  (* restart required for this setting to take effect)"))
			}

		default:
			// /config <key> <value> — set a config key.
			key := fields[1]
			value := strings.Join(fields[2:], " ")

			if err := config.SetField(session.settings, key, value); err != nil {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, err.Error()))
				return history
			}
			if err := config.Save(session.settings); err != nil {
				fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, "could not save settings: "+err.Error()))
				return history
			}

			// Find the def to decide hot vs restart-required.
			var def *config.ConfigDef
			for i := range config.ConfigDefs {
				if config.ConfigDefs[i].Key == key {
					def = &config.ConfigDefs[i]
					break
				}
			}
			if def != nil && def.Hot {
				if err := session.applyConfigField(ctx, key); err != nil {
					fmt.Fprintf(out, "%s\n", terminal.Yellow(session.colorLevel, "saved but could not hot-apply: "+err.Error()))
				} else {
					// For "mode", update the live prompt immediately.
					if key == "mode" {
						rl.SetPrompt(promptForMode(session.colorLevel, session.settings.Mode))
					}
					fmt.Fprintf(out, "%s = %s  %s\n",
						terminal.Bold(session.colorLevel, key),
						terminal.Cyan(session.colorLevel, value),
						terminal.Green(session.colorLevel, "✓ applied"),
					)
				}
			} else {
				fmt.Fprintf(out, "%s = %s  %s\n",
					terminal.Bold(session.colorLevel, key),
					terminal.Cyan(session.colorLevel, value),
					terminal.Dim(session.colorLevel, "(saved — restart required)"),
				)
			}
		}
		return history

	default:
		fmt.Fprintf(out, "%s\n", terminal.Red(session.colorLevel, fmt.Sprintf("unknown command %q; try /help", cmd)))
		return history
	}
}

// expandCmd implements the /cmd custom-command dispatcher. With no name it lists
// available commands (global + per-project) and returns run=false. With a name
// it loads the template, substitutes variables, and returns the expanded prompt
// with run=true so the caller submits it as a normal user turn. Unknown commands
// and empty expansions print to out and return run=false.
func (s *chatSession) expandCmd(out io.Writer, line string, history []ollama.Message) (string, bool) {
	fields := strings.Fields(line)

	globalDir, _ := config.CommandsDir()
	projectDir := filepath.Join(s.workingDir, ".braai", "commands")
	cmds := commands.Load(globalDir, projectDir)

	// "/cmd" alone: list available commands.
	if len(fields) < 2 {
		if len(cmds) == 0 {
			fmt.Fprintf(out, "No custom commands found.\nCreate *.md prompt templates in:\n  %s  (global)\n  %s  (this project)\n",
				globalDir, projectDir)
			return "", false
		}
		names := make([]string, 0, len(cmds))
		for n := range cmds {
			names = append(names, n)
		}
		sort.Strings(names)
		fmt.Fprintf(out, "Custom commands (%d):\n", len(cmds))
		for _, n := range names {
			c := cmds[n]
			header := "/cmd " + n
			if u := c.Usage(); u != "" {
				header += " " + u
			}
			fmt.Fprintf(out, "\n  %s  (%s)\n", terminal.Bold(s.colorLevel, header), c.Source)
			if c.Description != "" {
				fmt.Fprintf(out, "    %s\n", c.Description)
			}
		}
		return "", false
	}

	name := fields[1]
	c, ok := cmds[name]
	if !ok {
		fmt.Fprintf(out, "unknown command %q; type /cmd to list available commands\n", name)
		return "", false
	}
	prompt := c.Expand(fields[2:], lastAssistantMessage(history))
	if prompt == "" {
		fmt.Fprintf(out, "command %q produced an empty prompt\n", name)
		return "", false
	}
	return prompt, true
}

// lastAssistantMessage returns the text of the most recent assistant turn (used
// for the $SELECTION template variable), or "" if there is none.
func lastAssistantMessage(history []ollama.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "assistant" && history[i].Content != "" {
			return history[i].Content
		}
	}
	return ""
}

// printToolArgs renders a tool's JSON-Schema parameters (name, type, required
// flag, and description) for the `/tools full` listing. It is fully defensive:
// any missing or unexpectedly-typed schema field is skipped rather than causing
// a panic, so a malformed tool definition can never crash the REPL.
func printToolArgs(out io.Writer, colorLevel terminal.Level, params map[string]any) {
	props, _ := params["properties"].(map[string]any)
	if len(props) == 0 {
		fmt.Fprintln(out, "    arguments: (none)")
		return
	}

	// Collect the required parameter names (accept []string or []any, since the
	// schema is []string in code but becomes []any after a JSON round-trip).
	required := make(map[string]bool)
	switch req := params["required"].(type) {
	case []string:
		for _, name := range req {
			required[name] = true
		}
	case []any:
		for _, name := range req {
			if s, ok := name.(string); ok {
				required[s] = true
			}
		}
	}

	// Sort required arguments first, then alphabetically within each group.
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		if required[names[i]] != required[names[j]] {
			return required[names[i]]
		}
		return names[i] < names[j]
	})

	fmt.Fprintln(out, "    arguments:")
	for _, name := range names {
		spec, _ := props[name].(map[string]any)
		typ, _ := spec["type"].(string)
		if typ == "" {
			typ = "any"
		}
		if items, ok := spec["items"].(map[string]any); ok {
			if itemType, ok := items["type"].(string); ok {
				typ += "[" + itemType + "]"
			}
		}
		reqMark := ""
		if required[name] {
			reqMark = " (required)"
		}
		desc, _ := spec["description"].(string)

		fmt.Fprintf(out, "      %s %s%s\n", terminal.Bold(colorLevel, name), typ, reqMark)
		if desc != "" {
			fmt.Fprintf(out, "          %s\n", desc)
		}
	}
}

// formatTranscript formats the conversation history as a readable string,
// skipping the system prompt and internal tool-call plumbing.
func formatTranscript(history []ollama.Message) (string, error) {
	var b strings.Builder
	for _, m := range history {
		switch m.Role {
		case "user":
			fmt.Fprintf(&b, "## You\n\n%s\n\n", m.Content)
		case "assistant":
			if m.Content == "" {
				continue // tool-call-only turns have no user-facing text
			}
			fmt.Fprintf(&b, "## braai\n\n%s\n\n", m.Content)
		}
	}
	return b.String(), nil
}

// saveTranscript writes the user-visible parts of a conversation as readable Markdown.
// The path is resolved relative to the process's current directory, exactly like
// shell output redirection would — this is a user-initiated save of their own
// conversation, not something the model can trigger.
func saveTranscript(path string, history []ollama.Message) error {
	transcript, err := formatTranscript(history)
	if err != nil {
		return err
	}
	return os.WriteFile(path, []byte(transcript), 0o644)
}

// historyFilePath returns a per-user location for chat history persisted
// across sessions, best-effort (empty disables persistence rather than failing).
func historyFilePath() string {
	dir, err := config.Dir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "chat_history")
}

// expandTilde expands a leading "~" or "~/..." to the current user's home
// directory. Shells normally do this themselves, but a quoted argument
// (e.g. "~/some dir with spaces") suppresses tilde expansion, so a literal
// "~" can reach the program unexpanded — handle it here rather than making
// users get their quoting exactly right. "~user/..." (another user's home)
// is intentionally not supported, to keep this simple.
func expandTilde(path string) (string, error) {
	if path != "~" && !strings.HasPrefix(path, "~/") {
		return path, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("expand ~ in path: %w", err)
	}
	if path == "~" {
		return home, nil
	}
	return filepath.Join(home, path[2:]), nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// copyToClipboard copies text to the system clipboard using pbcopy (macOS) or xclip (Linux).
func copyToClipboard(text string) error {
	var cmd *exec.Cmd

	// Try pbcopy first (macOS)
	if _, err := exec.LookPath("pbcopy"); err == nil {
		cmd = exec.Command("pbcopy")
		cmd.Stdin = strings.NewReader(text)
		return cmd.Run()
	}

	// Try xclip (Linux)
	if _, err := exec.LookPath("xclip"); err == nil {
		cmd = exec.Command("xclip", "-selection", "clipboard")
		cmd.Stdin = strings.NewReader(text)
		return cmd.Run()
	}

	// Try xsel (Linux alternative)
	if _, err := exec.LookPath("xsel"); err == nil {
		cmd = exec.Command("xsel", "--clipboard", "--input")
		cmd.Stdin = strings.NewReader(text)
		return cmd.Run()
	}

	return fmt.Errorf("no clipboard utility found (tried pbcopy, xclip, xsel)")
}

// expandAtTokens scans line for @path tokens, resolves each against the
// working-directory root, reads the file content (plain text or extracted
// document text), and replaces each token inline with:
//
//	[Attached: <relpath>]
//	<content>
//
// Spaces in paths are escaped with a backslash, matching the standard shell
// convention (e.g. tab-completion on macOS produces this automatically):
//
//	@notes/Meeting\ Summary.md summarise this
//
// Unescaped spaces terminate the token, so bare @mention style text is
// left untouched. If any token cannot be resolved or read the function
// returns a non-nil error and the caller must abort the turn.
// A dim confirmation line is printed for each attachment.
func expandAtTokens(line string, session *chatSession, out io.Writer) (string, error) {
	if !strings.ContainsRune(line, '@') {
		return line, nil
	}

	type replacement struct {
		token   string // the original "@some/path" token as it appears in the line
		relPath string // the unescaped path (without leading @)
		content string
	}

	var replacements []replacement
	seen := map[string]bool{}

	// Scan left-to-right for '@' and collect the token character by character,
	// honouring backslash-escaped spaces so paths with spaces work naturally.
	i := 0
	runes := []rune(line)
	for i < len(runes) {
		if runes[i] != '@' {
			i++
			continue
		}
		// Consume characters after '@' until an unescaped space/tab or end-of-line.
		j := i + 1
		for j < len(runes) {
			if (runes[j] == ' ' || runes[j] == '\t') && (j == 0 || runes[j-1] != '\\') {
				break
			}
			j++
		}
		if j == i+1 {
			// bare '@' with nothing after it — skip
			i++
			continue
		}
		token := string(runes[i:j])                       // e.g. "@foo/bar\ baz.md"
		escaped := token[1:]                              // strip leading @
		relPath := strings.ReplaceAll(escaped, `\ `, " ") // unescape spaces

		if !seen[token] {
			seen[token] = true

			if _, err := session.root.Resolve(relPath); err != nil {
				return "", fmt.Errorf("@%s: %w", relPath, err)
			}
			content, err := session.registry.ReadAnyText(relPath)
			if err != nil {
				return "", fmt.Errorf("@%s: %w", relPath, err)
			}
			replacements = append(replacements, replacement{
				token:   token,
				relPath: relPath,
				content: content,
			})
		}
		i = j
	}

	if len(replacements) == 0 {
		return line, nil
	}

	// Replace each token inline and print a dim confirmation.
	result := line
	for _, r := range replacements {
		sizeKiB := float64(len(r.content)) / 1024
		fmt.Fprintf(out, "%s\n", terminal.Dim(session.colorLevel,
			fmt.Sprintf("  ⊕ %s (%.1f KiB)", r.relPath, sizeKiB)))
		block := fmt.Sprintf("[Attached: %s]\n%s", r.relPath, r.content)
		result = strings.ReplaceAll(result, r.token, block)
	}
	return result, nil
}

// braaiCompleter implements readline.AutoCompleter. It handles three cases:
//   - Current word starts with "/" and is the first token → complete slash-commands.
//   - Line starts with "/tree " and cursor is on the second token → complete root-level directory names.
//   - Current word starts with "@" → complete file paths in the working directory.
type braaiCompleter struct {
	root *security.Root
}

// slashCommands is the canonical list of completable slash-commands. Sub-command
// variants (e.g. "/tree hidden") are not listed separately — the user types the
// base command first, then adds sub-args by hand.
var slashCommands = []string{
	"/bye", "/cache", "/clear", "/cmd", "/config", "/copy",
	"/digest", "/export", "/forget-history", "/help",
	"/model", "/save", "/tools", "/tree",
}

// Do implements readline.AutoCompleter.
func (c *braaiCompleter) Do(line []rune, pos int) (newLine [][]rune, length int) {
	// Find the start of the current token: scan back from pos stopping at an
	// unescaped space/tab.
	wordStart := pos
	for wordStart > 0 {
		ch := line[wordStart-1]
		if (ch == ' ' || ch == '\t') && (wordStart < 2 || line[wordStart-2] != '\\') {
			break
		}
		wordStart--
	}
	word := string(line[wordStart:pos])

	// "/" completion — only when the word is the first token on the line.
	if strings.HasPrefix(word, "/") && wordStart == 0 {
		for _, cmd := range slashCommands {
			if strings.HasPrefix(cmd, word) {
				newLine = append(newLine, []rune(cmd[len(word):]))
			}
		}
		return newLine, len(word)
	}

	// "/tree" second-token completion — directory names, supporting sub-paths.
	// e.g. "/tree int"       → lists root dirs matching "int*"
	//      "/tree internal/" → lists dirs inside internal/
	//      "/tree internal/ag" → lists dirs inside internal/ matching "ag*"
	lineStr := string(line[:pos])
	if wordStart > 0 && strings.HasPrefix(lineStr, "/tree ") {
		prefix := strings.TrimPrefix(lineStr[:wordStart], "/tree ")
		// Only complete the immediately following token (no spaces in prefix).
		if !strings.Contains(strings.TrimSpace(prefix), " ") {
			// Split the word into a directory part and a basename prefix,
			// mirroring the same logic used in the /tree handler.
			partial := word // e.g. "internal/" or "internal/ag" or "int"
			treeDir := filepath.Dir(partial)
			treeBase := filepath.Base(partial)
			if treeDir == "." {
				treeDir = ""
			}
			if partial == "" || strings.HasSuffix(partial, "/") {
				treeDir = strings.TrimSuffix(partial, "/")
				treeBase = ""
			}

			searchDir := c.root.Abs()
			if treeDir != "" {
				searchDir = filepath.Join(searchDir, treeDir)
			}

			entries, err := os.ReadDir(searchDir)
			if err == nil {
				for _, e := range entries {
					if !e.IsDir() {
						continue
					}
					name := e.Name()
					if strings.HasPrefix(name, ".") {
						continue
					}
					if !strings.HasPrefix(strings.ToLower(name), strings.ToLower(treeBase)) {
						continue
					}
					// Return only the basename suffix — readline replaces len(treeBase)
					// chars before the cursor, leaving the already-typed dir prefix intact.
					newLine = append(newLine, []rune(name[len(treeBase):]))
				}
				return newLine, len(treeBase)
			}
		}
	}

	// "@" completion — file paths from the working directory.
	if !strings.HasPrefix(word, "@") {
		return nil, 0
	}

	// Unescape "\ " → " " to get the real partial path the user has typed so far.
	partial := strings.ReplaceAll(word[1:], `\ `, " ")

	// Split into directory prefix and file-name prefix.
	dir := filepath.Dir(partial)
	if dir == "." {
		dir = ""
	}
	base := filepath.Base(partial)
	if partial == "" || strings.HasSuffix(partial, "/") {
		dir = strings.TrimSuffix(partial, "/")
		base = ""
	}

	searchDir := c.root.Abs()
	if dir != "" {
		searchDir = filepath.Join(searchDir, dir)
	}

	entries, err := os.ReadDir(searchDir)
	if err != nil {
		return nil, 0
	}

	for _, e := range entries {
		name := e.Name()
		// Skip hidden files and known noisy dirs.
		if strings.HasPrefix(name, ".") {
			continue
		}
		if !strings.HasPrefix(strings.ToLower(name), strings.ToLower(base)) {
			continue
		}
		// Return only the suffix readline needs to append: the characters of
		// the escaped name that come after what the user has already typed
		// (len(base) unescaped chars). readline replaces len(base) chars before
		// the cursor with this suffix, producing the full name in-place.
		escapedName := strings.ReplaceAll(name, " ", `\ `)
		if e.IsDir() {
			escapedName += "/"
		}
		newLine = append(newLine, []rune(escapedName[len(base):]))
	}
	return newLine, len(base)
}

// citation holds one source reference extracted from a tool call.
type citation struct {
	relPath string // relative path as given to the tool
	line    int    // 0 means no line info
}

// displayText returns the human-readable label: "path:line" or "path".
func (c citation) displayText() string {
	if c.line > 0 {
		return fmt.Sprintf("%s:%d", c.relPath, c.line)
	}
	return c.relPath
}

// formatElapsed renders a turn's wall-clock duration as a short, human
// glance-able string: sub-10-second durations get one decimal place of
// precision (e.g. "3.2s"), since that's the range where the difference
// between e.g. 1s and 4s actually matters to the user; anything at or above
// 10s rounds to whole seconds so the display doesn't jitter across nearly
// every call.
func formatElapsed(d time.Duration) string {
	secs := d.Seconds()
	if secs < 10 {
		return fmt.Sprintf("%.1fs", secs)
	}
	return fmt.Sprintf("%.0fs", secs)
}

// extractCitations walks a slice of ToolCallRecords and returns a deduplicated,
// sorted slice of citations for every file-reading tool call made during a Run.
func extractCitations(calls []agent.ToolCallRecord) []citation {
	type key struct {
		path string
		line int
	}
	seen := map[key]bool{}
	var out []citation

	add := func(relPath string, line int) {
		if relPath == "" {
			return
		}
		k := key{relPath, line}
		if seen[k] {
			return
		}
		seen[k] = true
		out = append(out, citation{relPath: relPath, line: line})
	}

	for _, tc := range calls {
		switch tc.Name {
		case "read":
			// Single path with optional start_line.
			if p, ok := tc.Arguments["path"].(string); ok && p != "" {
				line := 0
				if sl, ok := tc.Arguments["start_line"].(float64); ok {
					line = int(sl)
				}
				add(p, line)
			}
			// Batch paths — no line info.
			if paths, ok := tc.Arguments["paths"].([]any); ok {
				for _, v := range paths {
					if p, ok := v.(string); ok {
						add(p, 0)
					}
				}
			}

		case "search":
			if len(tc.Result) == 0 {
				continue
			}
			var result struct {
				Matches []struct {
					Path string `json:"path"`
					Line int    `json:"line"` // 0 for semantic matches
				} `json:"matches"`
			}
			if err := json.Unmarshal(tc.Result, &result); err != nil {
				continue
			}
			for _, m := range result.Matches {
				add(m.Path, m.Line)
			}

		case "get_chunk", "stat_file":
			if p, ok := tc.Arguments["path"].(string); ok && p != "" {
				add(p, 0)
			}
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].relPath != out[j].relPath {
			return out[i].relPath < out[j].relPath
		}
		return out[i].line < out[j].line
	})
	return out
}

// citationLinks converts a []citation into display strings of the form
// [path:line] or [path], each enclosed in square brackets so filenames
// with spaces remain clearly delimited in the Sources footer.
func citationLinks(citations []citation) []string {
	out := make([]string, len(citations))
	for i, c := range citations {
		out[i] = "[" + c.displayText() + "]"
	}
	return out
}

// saveTranscriptJSON writes the user-visible parts of a conversation as a JSON
// array of {role, content} objects, skipping the system message, tool-only
// turns, and tool-role messages.
func saveTranscriptJSON(path string, history []ollama.Message) error {
	type jsonMsg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	var msgs []jsonMsg
	for _, m := range history {
		switch m.Role {
		case "user":
			msgs = append(msgs, jsonMsg{Role: "user", Content: m.Content})
		case "assistant":
			if m.Content == "" {
				continue // tool-call-only turns
			}
			msgs = append(msgs, jsonMsg{Role: "assistant", Content: m.Content})
		}
	}
	data, err := json.MarshalIndent(msgs, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// printTree renders the contents of dir as an ASCII tree, writing to out.
// prefix is the indentation string for the current level (built up recursively).
// showHidden controls whether dotfiles/dotdirs are included; entries in
// tools.SkipDirNames are always omitted regardless of showHidden.
func printTree(out io.Writer, dir string, prefix string, showHidden bool) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Fprintf(out, "%s[error reading directory: %v]\n", prefix, err)
		return
	}

	// Filter entries.
	var visible []os.DirEntry
	for _, e := range entries {
		name := e.Name()
		if tools.SkipDirNames[name] {
			continue
		}
		if !showHidden && strings.HasPrefix(name, ".") {
			continue
		}
		visible = append(visible, e)
	}

	for i, e := range visible {
		last := i == len(visible)-1
		connector := "├── "
		childPrefix := prefix + "│   "
		if last {
			connector = "└── "
			childPrefix = prefix + "    "
		}
		label := e.Name()
		if e.IsDir() {
			label += "/"
		}
		fmt.Fprintf(out, "%s%s%s\n", prefix, connector, label)
		if e.IsDir() {
			printTree(out, filepath.Join(dir, e.Name()), childPrefix, showHidden)
		}
	}
}
