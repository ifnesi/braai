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

	"github.com/chzyer/readline"

	"braai/internal/agent"
	"braai/internal/cache"
	"braai/internal/config"
	"braai/internal/ollama"
	"braai/internal/security"
	"braai/internal/staticembed"
	"braai/internal/terminal"
	"braai/internal/tools"
)

// version is the released version of braai, printed by --version.
const version = "0.0.6"

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
		ollamaHost    = fs.String("ollama-host", "", "Ollama server base URL (default http://localhost:11434, or ~/.braai/settings.json)")
		model         = fs.String("model", "", "Ollama model name to use (default: first model available on the server)")
		embedModel    = fs.String("embed-model", "", "Hugging Face repo of the static embedding model for semantic search. Default: minishlab/potion-retrieval-32M, or ~/.braai/settings.json")
		workingDir    = fs.String("working-dir", "", "Root directory the agent may inspect (default: current directory)")
		prompt        = fs.String("prompt", "", "Single prompt to run non-interactively. If omitted, starts an interactive chat, unless trailing args or stdin provide a prompt.")
		verbose       = fs.Bool("verbose", false, "Print tool calls and intermediate steps")
		hideReasoning = fs.Bool("hide-reasoning", false, "Don't stream the model's reasoning/thinking trace before its answer (shown by default, on models that support it)")
		maxToolCalls  = fs.Int("max-tool-calls", 0, "Maximum number of tool calls per request (default 100, or ~/.braai/settings.json)")
		maxReadBytes  = fs.Int("max-read-bytes", -1, "Maximum bytes read_file returns (-1 = no limit)")
		showVersion   = fs.Bool("version", false, "Print the braai version and exit")
		outputFormat  = fs.String("output", "text", `Output format: "text" (default, streamed to stdout as produced) or "json" (buffered; a single JSON object per answer with the answer, reasoning, and tool calls used)`)
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
		fmt.Fprintf(stdout, "braai %s\n", version)
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

	client := ollama.NewClient(host)

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
	// user-edited settings.json with the flag's zero default.
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
	if cacheMaxBytes <= 0 {
		cacheMaxBytes = 1 << 30 // 1 GiB
	}

	// Load the in-process static embedding model (downloaded from Hugging Face
	// on first use). Non-fatal: if it fails, chat still works and semantic
	// search reports a clear "unavailable" error instead of crashing.
	var embedModelObj *staticembed.Model
	if modelsDir, mdErr := config.ModelsDir(); mdErr == nil {
		if dir, dErr := staticembed.EnsureModel(ctx, resolvedEmbedModel, modelsDir); dErr == nil {
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

	limits := tools.DefaultLimits()
	limits.MaxReadBytes = *maxReadBytes

	// In JSON mode, output is buffered and printed once as a single object
	// rather than streamed, so the agent's streaming writer is discarded.
	agentStdout := stdout
	colorLevel := terminal.Detect(stdout)
	if jsonOutput {
		agentStdout = io.Discard
		colorLevel = terminal.None
	}

	session := newChatSession(client, root, limits, settings, resolvedEmbedModel, embedModelObj, resolvedMaxToolCalls, *verbose, !*hideReasoning, dir, agentStdout, colorLevel, stderr, semanticCache)
	if err := session.switchModel(ctx, selectedModel); err != nil {
		return err
	}

	trailing := strings.TrimSpace(strings.Join(fs.Args(), " "))
	initialPrompt := firstNonEmpty(*prompt, trailing)

	if initialPrompt == "" && !terminal.IsTerminal(stdin) {
		// stdin is piped and no explicit prompt/trailing args given: treat all of stdin as the prompt.
		data, readErr := io.ReadAll(stdin)
		if readErr != nil {
			return fmt.Errorf("read stdin: %w", readErr)
		}
		initialPrompt = strings.TrimSpace(string(data))
	}

	if initialPrompt != "" {
		return runOnce(ctx, session.ag, initialPrompt, stdout, jsonOutput)
	}

	return runChat(ctx, session, jsonOutput)
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
}

func newChatSession(client *ollama.Client, root *security.Root, limits tools.Limits, settings *config.Settings, embedModel string, embedder *staticembed.Model, maxToolCalls int, verbose, showReasoning bool, workingDir string, stdout io.Writer, colorLevel terminal.Level, verboseWriter io.Writer, semanticCache *cache.Cache) *chatSession {
	return &chatSession{
		client:        client,
		root:          root,
		limits:        limits,
		settings:      settings,
		embedModel:    embedModel,
		embedder:      embedder,
		maxToolCalls:  maxToolCalls,
		verbose:       verbose,
		showReasoning: showReasoning,
		workingDir:    workingDir,
		stdout:        stdout,
		colorLevel:    colorLevel,
		verboseWriter: verboseWriter,
		cache:         semanticCache,
	}
}

// switchModel rebuilds the tool registry and agent for model (re-checking
// vision support and context length, since both are model-specific) and
// persists it to ~/.braai/settings.json as the new default.
func (s *chatSession) switchModel(ctx context.Context, model string) error {
	info, err := s.client.ShowModel(ctx, model)
	if err != nil && s.verbose {
		fmt.Fprintf(s.verboseWriter, "warning: could not check capabilities for %s, assuming no vision support and no context-length warnings: %v\n", model, err)
	}

	var embedClient tools.Embedder
	if s.embedder != nil {
		embedClient = s.embedder
	}
	registry := tools.NewRegistry(s.root, s.limits, info.HasCapability("vision"), embedClient, s.embedModel)
	registry.SetCache(s.cache)
	ag := agent.New(s.client, registry, agent.Options{
		Model:         model,
		MaxToolCalls:  s.maxToolCalls,
		Verbose:       s.verbose,
		VerboseWriter: s.verboseWriter,
		ShowReasoning: s.showReasoning,
		Stdout:        s.stdout,
		ColorLevel:    s.colorLevel,
		ContextLength: info.ContextLength,
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
func runOnce(ctx context.Context, ag *agent.Agent, prompt string, stdout io.Writer, jsonOutput bool) error {
	history := []ollama.Message{
		agent.SystemMessage(),
		{Role: "user", Content: prompt},
	}
	result, err := ag.Run(ctx, history)
	if err != nil {
		return err
	}
	if jsonOutput {
		return printJSONResult(stdout, result)
	}
	return nil
}

// printJSONResult encodes an agent.RunResult as a single indented JSON
// object (answer, reasoning if any, and the tool calls used to produce it).
func printJSONResult(w io.Writer, result agent.RunResult) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

// maxHistoryEntries caps how many lines are kept in ~/.braai/chat_history.
// Readline trims the on-disk file to this limit itself as soon as it opens
// it, so the file is adjusted every time braai starts, not just on save.
const maxHistoryEntries = 100

// runChat drives an interactive prompt with readline-style line editing:
// left/right arrows, Ctrl-A/Ctrl-E to jump to the start/end of the line, and
// Ctrl-C to clear the current input instead of killing the process (Ctrl + d
// or 'exit'/'quit' still leave the chat). A few slash-commands are also
// available: /clear, /tools, /save <file>, /help. When jsonOutput is set,
// each answer is printed as a buffered JSON object instead of streamed text.
func runChat(ctx context.Context, session *chatSession, jsonOutput bool) error {
	rl, err := readline.NewEx(&readline.Config{
		Prompt:       ">>> ",
		HistoryFile:  historyFilePath(),
		HistoryLimit: maxHistoryEntries,
	})
	if err != nil {
		return fmt.Errorf("start interactive prompt: %w", err)
	}
	defer rl.Close()

	fmt.Fprintf(rl.Stdout(), "braai %s\nWorking Directory %s\nInteractive chat using model %s.\nUse Ctrl + d or /bye to exit, or /help for commands.\n\n", version, session.workingDir, session.model)

	history := []ollama.Message{agent.SystemMessage()}
	for {
		line, err := rl.Readline()
		if errors.Is(err, readline.ErrInterrupt) {
			// Ctrl-C: show exit hint and reprompt rather than exiting.
			fmt.Fprintf(rl.Stdout(), "\nUse Ctrl + d or /bye to exit.\n")
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
		if line == "exit" || line == "quit" || line == "/bye" {
			return nil
		}

		if strings.HasPrefix(line, "/") {
			if line == "/bye" {
				return nil
			}
			history = handleSlashCommand(ctx, rl, line, history, session)
			continue
		}

		history = append(history, ollama.Message{Role: "user", Content: line})
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
		go func() {
			select {
			case <-sigChan:
				cancel()
			case <-turnDone:
				// Run finished normally; exit without leaking this goroutine.
			}
		}()

		result, err := session.ag.Run(runCtx, history)
		signal.Stop(sigChan)
		close(turnDone) // unblock the watcher goroutine
		cancel()

		if err != nil {
			sp.Stop() // erase spinner before printing error
			if errors.Is(err, context.Canceled) {
				fmt.Fprintf(rl.Stdout(), "\n")
				continue
			}
			fmt.Fprintf(rl.Stdout(), "%s\n", terminal.Red(session.colorLevel, "error: "+err.Error()))
			continue
		}
		if jsonOutput {
			if err := printJSONResult(rl.Stdout(), result); err != nil {
				fmt.Fprintf(rl.Stdout(), "%s\n", terminal.Red(session.colorLevel, "error encoding JSON output: "+err.Error()))
			}
		}
		history = result.History
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
		fmt.Fprintln(out, "Commands:")
		fmt.Fprintf(out, "  %s%s reset the conversation history\n", terminal.Bold(session.colorLevel, "/clear"), strings.Repeat(" ", 18-len("/clear")))
		fmt.Fprintf(out, "  %s%s erase ~/.braai/chat_history (the up/down recall history)\n", terminal.Bold(session.colorLevel, "/forget-history"), strings.Repeat(" ", 18-len("/forget-history")))
		fmt.Fprintf(out, "  %s%s list tools available to the model (/tools full for details)\n", terminal.Bold(session.colorLevel, "/tools"), strings.Repeat(" ", 18-len("/tools")))
		fmt.Fprintf(out, "  %s%s show the current model and list models available on the server\n", terminal.Bold(session.colorLevel, "/model"), strings.Repeat(" ", 18-len("/model")))
		fmt.Fprintf(out, "  %s%s switch to a different model and save it as the default\n", terminal.Bold(session.colorLevel, "/model <name>"), strings.Repeat(" ", 18-len("/model <name>")))
		fmt.Fprintf(out, "  %s%s save the conversation transcript to a file\n", terminal.Bold(session.colorLevel, "/save <file>"), strings.Repeat(" ", 18-len("/save <file>")))
		fmt.Fprintf(out, "  %s%s copy the last answer to clipboard\n", terminal.Bold(session.colorLevel, "/copy"), strings.Repeat(" ", 18-len("/copy")))
		fmt.Fprintf(out, "  %s%s show or clear the semantic-search cache (/cache clear)\n", terminal.Bold(session.colorLevel, "/cache"), strings.Repeat(" ", 18-len("/cache")))
		fmt.Fprintf(out, "  %s%s show this message\n", terminal.Bold(session.colorLevel, "/help"), strings.Repeat(" ", 18-len("/help")))
		fmt.Fprintf(out, "  %s%s leave the chat (Ctrl + d, also works)\n", terminal.Bold(session.colorLevel, "/bye, exit, quit"), strings.Repeat(" ", 18-len("/bye, exit, quit")))
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
		return []ollama.Message{agent.SystemMessage()}

	case "/forget-history":
		// This is the persisted up/down recall history (~/.braai/chat_history),
		// distinct from /clear's conversation context — wipe both the on-disk
		// file and the in-memory copy readline is holding for this session.
		path := historyFilePath()
		if path != "" {
			if err := os.WriteFile(path, nil, 0o644); err != nil {
				fmt.Fprintf(out, "could not erase %s: %v\n", path, err)
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
			fmt.Fprintf(out, "could not list models from ollama: %v\n", err)
			return history
		}

		if len(fields) == 1 {
			fmt.Fprintf(out, "Current model: %s\n", session.model)
			fmt.Fprintln(out, "Available models:")
			for _, m := range available {
				marker := "  "
				if m == session.model {
					marker = "* "
				}
				fmt.Fprintf(out, "%s%s\n", marker, m)
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
			fmt.Fprintf(out, "unknown model %q. Available models:\n", target)
			for _, m := range available {
				fmt.Fprintf(out, "  %s\n", m)
			}
			return history
		}

		if err := session.switchModel(ctx, target); err != nil {
			fmt.Fprintf(out, "could not switch to model %q: %v\n", target, err)
			return history
		}
		fmt.Fprintf(out, "Switched to model %s (saved as default).\n", target)
		return history

	case "/save":
		if len(fields) < 2 {
			fmt.Fprintln(out, "usage: /save <file>")
			return history
		}
		if err := saveTranscript(fields[1], history); err != nil {
			fmt.Fprintf(out, "could not save transcript: %v\n", err)
		} else {
			fmt.Fprintf(out, "saved conversation to %s\n", fields[1])
		}
		return history

	case "/copy":
		transcript, err := formatTranscript(history)
		if err != nil {
			fmt.Fprintf(out, "could not format transcript: %v\n", err)
			return history
		}
		if err := copyToClipboard(transcript); err != nil {
			fmt.Fprintf(out, "could not copy to clipboard: %v\n", err)
		} else {
			fmt.Fprintln(out, "conversation copied to clipboard")
		}
		return history

	case "/cache":
		if session.cache == nil {
			fmt.Fprintln(out, "semantic cache is disabled")
			return history
		}
		if len(fields) >= 2 && fields[1] == "clear" {
			if err := session.cache.Clear(); err != nil {
				fmt.Fprintf(out, "could not clear cache: %v\n", err)
			} else {
				fmt.Fprintln(out, "semantic cache cleared")
			}
			return history
		}
		files, chunks, bytes := session.cache.Status()
		fmt.Fprintf(out, "semantic cache: %d files, %d chunks, %.1f MiB (use /cache clear to wipe)\n",
			files, chunks, float64(bytes)/(1024*1024))
		return history

	default:
		fmt.Fprintf(out, "unknown command %q; try /help\n", cmd)
		return history
	}
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

// saveTranscript writes the user-visible parts of a conversation (skipping
// the system prompt and internal tool-call plumbing) as readable Markdown.
// The path is resolved relative to the process's current directory, exactly
// like shell output redirection would — this is a user-initiated save of
// their own conversation, not something the model can trigger.
// formatTranscript formats the conversation history as a readable string.
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
	return dir + "/chat_history"
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
