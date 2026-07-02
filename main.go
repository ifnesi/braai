// Command braai is a read-only AI agent over a local working directory,
// using a local Ollama server for reasoning and tool selection.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/chzyer/readline"

	"braai/internal/agent"
	"braai/internal/config"
	"braai/internal/ollama"
	"braai/internal/security"
	"braai/internal/tools"
)

// version is the released version of braai, printed by --version.
const version = "0.0.1"

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
		ollamaURL     = fs.String("ollama-url", "", "Alias for --ollama-host")
		model         = fs.String("model", "", "Ollama model name to use (default: first model available on the server)")
		workingDir    = fs.String("working-dir", "", "Root directory the agent may inspect (default: current directory)")
		prompt        = fs.String("prompt", "", "Single prompt to run non-interactively. If omitted, starts an interactive chat, unless trailing args or stdin provide a prompt.")
		verbose       = fs.Bool("verbose", false, "Print tool calls and intermediate steps")
		showReasoning = fs.Bool("show-reasoning", false, "Stream the model's reasoning/thinking trace before its answer, on models that support it")
		maxToolCalls  = fs.Int("max-tool-calls", 8, "Maximum number of tool calls per request")
		maxReadBytes  = fs.Int("max-read-bytes", -1, "Maximum bytes read_file returns (-1 = no limit)")
		showVersion   = fs.Bool("version", false, "Print the braai version and exit")
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
		return err
	}

	if *showVersion {
		fmt.Fprintf(stdout, "braai %s\n", version)
		return nil
	}

	settings, err := config.Load()
	if err != nil {
		return fmt.Errorf("load settings: %w", err)
	}

	host := firstNonEmpty(*ollamaHost, *ollamaURL, settings.OllamaHost, "http://localhost:11434")
	dir := *workingDir
	if dir == "" {
		dir = "."
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

	// Persist the last-used host/model as defaults for next time.
	settings.OllamaHost = host
	settings.Model = selectedModel
	settings.MaxToolCall = *maxToolCalls
	if err := config.Save(settings); err != nil && *verbose {
		fmt.Fprintf(stderr, "warning: could not save settings: %v\n", err)
	}

	limits := tools.DefaultLimits()
	limits.MaxReadBytes = *maxReadBytes
	registry := tools.NewRegistry(root, limits)

	ag := agent.New(client, registry, agent.Options{
		Model:         selectedModel,
		MaxToolCalls:  *maxToolCalls,
		Verbose:       *verbose,
		VerboseWriter: stderr,
		ShowReasoning: *showReasoning,
		Stdout:        stdout,
	})

	trailing := strings.TrimSpace(strings.Join(fs.Args(), " "))
	initialPrompt := firstNonEmpty(*prompt, trailing)

	if initialPrompt == "" && !isInteractive(stdin) {
		// stdin is piped and no explicit prompt/trailing args given: treat all of stdin as the prompt.
		data, readErr := io.ReadAll(stdin)
		if readErr != nil {
			return fmt.Errorf("read stdin: %w", readErr)
		}
		initialPrompt = strings.TrimSpace(string(data))
	}

	if initialPrompt != "" {
		return runOnce(ctx, ag, initialPrompt, stdout)
	}

	return runChat(ctx, ag)
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

// runOnce executes a single prompt. The answer is streamed directly to
// stdout by ag.Run as it arrives, so the returned text is not printed again.
func runOnce(ctx context.Context, ag *agent.Agent, prompt string, stdout io.Writer) error {
	history := []ollama.Message{
		agent.SystemMessage(),
		{Role: "user", Content: prompt},
	}
	_, _, err := ag.Run(ctx, history)
	return err
}

// maxHistoryEntries caps how many lines are kept in ~/.braai/chat_history.
// Readline trims the on-disk file to this limit itself as soon as it opens
// it, so the file is adjusted every time braai starts, not just on save.
const maxHistoryEntries = 100

// runChat drives an interactive prompt with readline-style line editing:
// left/right arrows, Ctrl-A/Ctrl-E to jump to the start/end of the line, and
// Ctrl-C to clear the current input instead of killing the process (Ctrl-D
// or 'exit'/'quit' still leave the chat).
func runChat(ctx context.Context, ag *agent.Agent) error {
	rl, err := readline.NewEx(&readline.Config{
		Prompt:       "> ",
		HistoryFile:  historyFilePath(),
		HistoryLimit: maxHistoryEntries,
	})
	if err != nil {
		return fmt.Errorf("start interactive prompt: %w", err)
	}
	defer rl.Close()

	fmt.Fprintln(rl.Stdout(), "braai interactive chat. Type your question, or 'exit'/'quit' to leave. Ctrl-D also exits.")

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

		history = append(history, ollama.Message{Role: "user", Content: line})
		// The answer streams straight to stdout as it arrives; only the
		// updated history needs to be captured here.
		_, updated, err := ag.Run(ctx, history)
		if err != nil {
			fmt.Fprintf(rl.Stdout(), "error: %v\n", err)
			continue
		}
		history = updated
	}
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

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func isInteractive(r io.Reader) bool {
	f, ok := r.(*os.File)
	if !ok {
		return true
	}
	stat, err := f.Stat()
	if err != nil {
		return true
	}
	return (stat.Mode() & os.ModeCharDevice) != 0
}
