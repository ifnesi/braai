# braai

`braai` is a small, read-only AI agent CLI that answers questions about a local
working directory. It uses a local [Ollama](https://ollama.com) server for
reasoning and lets the model call a fixed set of read-only filesystem tools
(list, read, search by name, search by content, stat) to gather evidence
before answering. It never writes, deletes, or executes anything on your
filesystem.

## Requirements

- Go 1.21+
- A running local Ollama server (`ollama serve`) with at least one model
  pulled (e.g. `ollama pull llama3.1`). The model should support Ollama's
  native tool calling for best results (check with `ollama show <model>` —
  look for `tools` under capabilities).

## Build

```sh
go build -o braai .
```

### Build, test, and install system-wide

To build, run the test suite, and install `braai` onto your `PATH` so it can
be called from any directory (like `ls` or `grep`), use the provided script:

```sh
./build.sh
```

This runs `go test ./...` and `go vet ./...`, then builds and moves the
binary to `~/.local/bin/braai` (created if it doesn't exist). Most modern
shells already have `~/.local/bin` on `PATH`; if not, the script prints the
line to add to your shell profile. Override the install location with:

```sh
INSTALL_DIR=/usr/local/bin ./build.sh
```

Once installed, run it from anywhere:

```sh
cd ~/some/other/project
braai "what does this project do?"
```

## Usage

```sh
# One-shot prompt as a trailing argument
./braai --model llama3.1 "summarize the architecture in this repo"

# Restrict to a subdirectory
./braai --model qwen3 --working-dir ./docs "find mentions of Kafka and summarize them"

# Pipe a question via stdin
cat question.txt | ./braai --model mistral --working-dir .

# Explicit --prompt flag works the same as a trailing argument
./braai --model llama3.1 --prompt "what does main.go do?"

# Omit prompt/stdin entirely to start an interactive chat
./braai --model llama3.1

# Stream the model's reasoning/thinking trace before its answer
./braai --model llama3.1 --show-reasoning "why does main.go split runOnce and runChat?"
```

Answers are always streamed to stdout as the model produces them, rather than
printed only once the full response is ready.

The interactive chat supports standard readline-style line editing: left/right
arrows to move within the line, Ctrl-A/Ctrl-E to jump to the start/end of the
line, Ctrl-C to clear the current input (without exiting), Ctrl-D or
`exit`/`quit` to leave, and up/down arrows to recall history from previous
sessions (persisted to `~/.braai/chat_history`, capped at the last 100
entries — the file is trimmed to that limit every time braai starts).

If `--model` is omitted, `braai` uses the first model reported by the Ollama
server (preferring the last model you used, if it's still installed). If no
models are installed at all, it exits with an error instead of starting.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | first available | Ollama model to use |
| `--working-dir` | `.` | Root directory the agent may inspect |
| `--ollama-host` / `--ollama-url` | `http://localhost:11434` | Ollama server base URL |
| `--prompt` | — | Run one prompt non-interactively (trailing args work the same way) |
| `--verbose` | `false` | Print tool calls and intermediate steps to stderr |
| `--show-reasoning` | `false` | Stream the model's reasoning/thinking trace (on models that support it) before its answer |
| `--max-tool-calls` | `8` | Max tool calls allowed per request before aborting |
| `--max-read-bytes` | `-1` (no limit) | Max bytes `read_file` will return |
| `--version` | — | Print the braai version and exit |

### Configuration file

`braai` persists your last-used Ollama host and model to
`~/.braai/settings.json` so subsequent runs can reuse them as defaults.
Command-line flags always take precedence over this file.

## Read-only toolset

The agent can only call these five tools, all confined to `--working-dir`:

- **list_dir** — list entries in a directory, with optional recursion depth.
- **read_file** — read a text file, with optional line ranges and a
  configurable max-bytes truncation. Refuses binary files.
- **search_name** — case-insensitive (by default) substring search over file
  and directory names.
- **search_content** — plain-text search over file contents, returning file
  path, line number, and a short excerpt per match. Skips binary and
  oversized files.
- **stat_file** — metadata: type, size, modification time, permissions,
  extension.

There are no write, delete, rename, mkdir, chmod, or shell-execution tools —
this is intentional and hardcoded, not something that can be enabled.

`.git`, `node_modules`, `vendor`, `.idea`, and `.DS_Store` are skipped by
directory listing, name search, and content search for usability (see
`internal/tools/tools.go`).

## Security model

All tool paths are resolved through `internal/security`, which:

- Resolves the working directory to an absolute, symlink-resolved path once
  at startup (the "root").
- Joins every tool-supplied relative path against the root, cleans it, and
  rejects anything that resolves outside the root (including `..` traversal
  and absolute-path inputs).
- Re-resolves symlinks for paths that exist on disk and re-checks
  containment, so a symlink inside the working directory cannot be used to
  read a file outside it.

No tool ever opens a file for writing, and there is no command-execution
tool, so there is no path by which the agent can modify your filesystem.

## Architecture

```
main.go                    CLI flag parsing, stdin/interactive wiring
internal/agent/agent.go    Chat + tool-calling loop, system prompt
internal/ollama/client.go  Minimal /api/chat and /api/tags HTTP client
internal/tools/            The five read-only tools + their JSON schemas
internal/security/path.go  Path confinement/validation helpers
internal/config/config.go  ~/.braai/settings.json persistence
```

The agent loop (`internal/agent/agent.go`) is intentionally simple:

1. Send the running message history plus the tool schemas to `/api/chat`.
2. If the model's response includes tool calls, execute each one against the
   `tools.Registry` (which enforces the working-directory confinement) and
   append the results as `tool` role messages.
3. Repeat until the model returns a plain text answer or a configurable
   `--max-tool-calls` guardrail is hit.

This relies on Ollama's native OpenAI-compatible tool-calling support in
`/api/chat` (message `tool_calls` / role `"tool"`). The `ollama.Message` and
`ollama.Tool` types are a thin, self-contained representation of that wire
format, so a JSON-emission fallback for models without native tool support
could be added later as an alternate implementation behind the same
`agent.Agent` interface without touching the CLI or tools layer.

## Testing

```sh
go test ./...
```

Unit tests cover path validation (traversal, absolute paths, symlink escape)
in `internal/security` and tool behavior (binary refusal, content/name
search, stat, directory listing) in `internal/tools`.
