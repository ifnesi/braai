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
	"strings"
	"syscall"

	"github.com/chzyer/readline"

	"braai/internal/agent"
	"braai/internal/config"
	"braai/internal/ollama"
	"braai/internal/security"
	"braai/internal/terminal"
	"braai/internal/tools"
)

// version is the released version of braai, printed by --version.
const version = "0.0.4"

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
		workingDir    = fs.String("working-dir", "", "Root directory the agent may inspect (default: current directory)")
		prompt        = fs.String("prompt", "", "Single prompt to run non-interactively. If omitted, starts an interactive chat, unless trailing args or stdin provide a prompt.")
		verbose       = fs.Bool("verbose", false, "Print tool calls and intermediate steps")
		hideReasoning = fs.Bool("hide-reasoning", false, "Don't stream the model's reasoning/thinking trace before its answer (shown by default, on models that support it)")
		maxToolCalls  = fs.Int("max-tool-calls", agent.DefaultMaxToolCalls, "Maximum number of tool calls per request")
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

	// Record the last-used host and tool-call limit as defaults for next
	// time; the model itself is persisted by chatSession.switchModel below
	// (also used at runtime by /model), so it isn't set here.
	settings.OllamaHost = host
	settings.MaxToolCalls = *maxToolCalls

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

	session := newChatSession(client, root, limits, settings, *maxToolCalls, *verbose, !*hideReasoning, dir, agentStdout, colorLevel, stderr)
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
}

func newChatSession(client *ollama.Client, root *security.Root, limits tools.Limits, settings *config.Settings, maxToolCalls int, verbose, showReasoning bool, workingDir string, stdout io.Writer, colorLevel terminal.Level, verboseWriter io.Writer) *chatSession {
	return &chatSession{
		client:        client,
		root:          root,
		limits:        limits,
		settings:      settings,
		maxToolCalls:  maxToolCalls,
		verbose:       verbose,
		showReasoning: showReasoning,
		workingDir:    workingDir,
		stdout:        stdout,
		colorLevel:    colorLevel,
		verboseWriter: verboseWriter,
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

	registry := tools.NewRegistry(s.root, s.limits, info.HasCapability("vision"), s.client, model)
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
// Ctrl-C to clear the current input instead of killing the process (Ctrl-D
// or 'exit'/'quit' still leave the chat). A few slash-commands are also
// available: /clear, /tools, /save <file>, /help. When jsonOutput is set,
// each answer is printed as a buffered JSON object instead of streamed text.
func runChat(ctx context.Context, session *chatSession, jsonOutput bool) error {
	rl, err := readline.NewEx(&readline.Config{
		Prompt:       "> ",
		HistoryFile:  historyFilePath(),
		HistoryLimit: maxHistoryEntries,
	})
	if err != nil {
		return fmt.Errorf("start interactive prompt: %w", err)
	}
	defer rl.Close()

	fmt.Fprintf(rl.Stdout(), "braai %s\nWorking Directory %s\nInteractive chat using model %s.\nType your question, 'exit'/'quit' or Ctrl-D to leave, or /help for commands.\n", version, session.workingDir, session.model)

	history := []ollama.Message{agent.SystemMessage()}
	for {
		line, err := rl.Readline()
		if errors.Is(err, readline.ErrInterrupt) {
			// Ctrl-C: clear the current line and reprompt rather than exiting.
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
		if line == "exit" || line == "quit" {
			return nil
		}

		if strings.HasPrefix(line, "/") {
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

		// Create a cancellable context for this agent run so Ctrl-C can interrupt it
		runCtx, cancel := context.WithCancel(ctx)
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT)
		go func() {
			<-sigChan
			cancel()
		}()

		result, err := session.ag.Run(runCtx, history)
		signal.Stop(sigChan)
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
		fmt.Fprintln(out, "  /clear             reset the conversation history")
		fmt.Fprintln(out, "  /forget-history    erase ~/.braai/chat_history (the up/down recall history)")
		fmt.Fprintln(out, "  /tools             list the tools available to the model")
		fmt.Fprintln(out, "  /model             show the current model and list models available on the server")
		fmt.Fprintln(out, "  /model <name>      switch to a different model and save it as the default")
		fmt.Fprintln(out, "  /save <file>       save the conversation transcript to a file")
		fmt.Fprintln(out, "  /copy              copy the last answer to clipboard")
		fmt.Fprintln(out, "  /help              show this message")
		fmt.Fprintln(out, "  exit, quit         leave the chat (Ctrl-D also works)")
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
		fmt.Fprintln(out, "Available tools:")
		for _, t := range session.registry.Definitions() {
			fmt.Fprintf(out, "  %-16s %s\n", t.Function.Name, t.Function.Description)
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

	default:
		fmt.Fprintf(out, "unknown command %q; try /help\n", cmd)
		return history
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
