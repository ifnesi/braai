# braai

`braai` is a small, read-only AI agent CLI that answers questions about a local
working directory. It uses a local [Ollama](https://ollama.com) server for
reasoning and lets the model call a fixed set of read-only filesystem tools
(list, read, batch-read, search by name, search by content, stat, and —
on vision-capable models — read/OCR images) to gather evidence before
answering. It never writes, deletes, or executes anything on your filesystem.

It works well as a local research assistant over a folder of meeting notes and
audio transcripts, e.g. `braai --working-dir ~/notes/2026-Q3 "summarize this
week's meetings and flag any action items"` — see
[Read-only toolset](#read-only-toolset) for the tools it uses to do that.

![Image](img/braai.png)

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

# Get a single structured JSON object instead of streamed text, e.g. for scripting
./braai --model llama3.1 --output json "list the files in internal/tools" | jq .
```

By default (`--output text`), answers are streamed to stdout as the model
produces them, rather than printed only once the full response is ready.
When `--show-reasoning` is used in a real terminal, the reasoning trace is
dimmed so it's visually distinct from the final answer; the dimming is
skipped automatically when stdout is piped or redirected, so redirected
output stays free of ANSI escape codes.

With `--output json`, streaming is disabled and braai instead prints one
JSON object per answer once it's complete:
`{"answer": "...", "reasoning": "...", "tool_calls": [{"name": "...", "arguments": {...}, "result": "..."}]}`.
`reasoning` is only populated when `--show-reasoning` is also set, and
`tool_calls` is omitted if the model answered without using any tools. This
works in both one-shot and interactive chat mode (one JSON object per turn).

The interactive chat supports standard readline-style line editing: left/right
arrows to move within the line, Ctrl-A/Ctrl-E to jump to the start/end of the
line, Ctrl-C to clear the current input (without exiting), Ctrl-D or
`exit`/`quit` to leave, and up/down arrows to recall history from previous
sessions (persisted to `~/.braai/chat_history`, capped at the last 100
entries — the file is trimmed to that limit every time braai starts).

A few slash-commands are available inside the chat:

| Command | Effect |
|---|---|
| `/help` | List available commands |
| `/clear` | Reset the conversation history and clear the visible screen (start fresh without restarting) |
| `/forget-history` | Erase `~/.braai/chat_history` — the up/down arrow recall history — separate from the conversation itself |
| `/tools` | List the tools currently available to the model |
| `/model` | Show the current model and list every model available on the Ollama server |
| `/model <name>` | Switch to a different model and save it as the default (persisted to `~/.braai/settings.json`) |
| `/save <file>` | Save the visible conversation (your messages + braai's answers) as Markdown |

If the conversation is getting close to the model's context window, braai
prints a warning (e.g. `warning: conversation is ~85% of gemma4:e4b's
estimated 131072-token context window...`) suggesting `/clear`, a shorter
prompt, or reading fewer files at once. This is based on a rough
character-count estimate, not the model's actual tokenizer, so treat it as a
heads-up rather than an exact measurement.

If `--model` is omitted, `braai` uses the first model reported by the Ollama
server (preferring the last model you used, if it's still installed). If no
models are installed at all, it exits with an error instead of starting.

On every startup, braai prints which model it's using (`using model:
<name>`) to stderr — so it never pollutes `--output json` or piped stdout —
and interactive chat also shows it in the opening banner. You can switch
models at any time from inside the chat with `/model` (see below) instead of
restarting with a different `--model` flag.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | first available | Ollama model to use |
| `--working-dir` | `.` | Root directory the agent may inspect |
| `--ollama-host` | `http://localhost:11434` | Ollama server base URL |
| `--prompt` | — | Run one prompt non-interactively (trailing args work the same way) |
| `--verbose` | `false` | Print tool calls and intermediate steps to stderr |
| `--show-reasoning` | `false` | Stream the model's reasoning/thinking trace (on models that support it) before its answer |
| `--max-tool-calls` | `100` | Max tool calls allowed per request before aborting |
| `--max-read-bytes` | `-1` (no limit) | Max bytes `read_file` will return |
| `--version` | — | Print the braai version and exit |
| `--output` | `text` | `text` streams the answer as produced; `json` buffers and prints one JSON object per answer (`answer`, `reasoning`, `tool_calls`) |

### Configuration file

`braai` persists your last-used Ollama host and model to
`~/.braai/settings.json` so subsequent runs can reuse them as defaults.
Command-line flags always take precedence over this file.

## Read-only toolset

The agent can only call the tools below, all confined to `--working-dir`:

- **list_dir** — list entries in a directory. Supports recursion depth, an
  `extensions` filter (e.g. only `.md`/`.txt`), and `sort_by: modified_time`
  to surface the most recently changed files first (handy for "find this
  week's meeting notes").
- **read_file** — read a text file, with optional line ranges and a
  configurable max-bytes truncation. Refuses binary files.
- **read_files** — read several text files in a single call (e.g. a batch of
  meeting notes to summarize together), instead of one `read_file` call per
  file. Capped at 20 files per call; per-file errors are reported inline
  without failing the whole batch.
- **search_name** — case-insensitive (by default) substring search over file
  and directory names, with an optional `extensions` filter.
- **search_content** — plain-text search over file contents, returning file
  path, line number, and a short excerpt per match. Skips binary and
  oversized files.
- **search_semantic** — search files by *meaning* rather than exact text,
  using Ollama embeddings (e.g. "find notes about the pricing decision" even
  if those exact words never appear). Ranks whole files by cosine similarity
  to the query. This is a brute-force, all-in-memory implementation: no
  vector database, no persistence — embeddings for files are computed lazily
  on first use and cached in memory (keyed by path + mtime) for the lifetime
  of the process, so repeated searches in one chat session don't recompute
  them. Requires the Ollama server to actually support embeddings (some
  builds need to be started with an embeddings flag); if it doesn't, the
  tool returns a clear error and the model is expected to fall back to
  `search_content`. Slower and coarser-grained (whole-file, not per-chunk)
  than `search_content`, so prefer `search_content` for known substrings.
- **stat_file** — metadata: type, size, modification time, permissions,
  extension.
- **read_image** — read a PNG/JPG/JPEG/GIF/WEBP and attach it to the
  conversation for the model to visually inspect (OCR text, describe a
  diagram or screenshot, etc.). **Only advertised to the model when the
  active model reports `vision` in its Ollama capabilities** (check with
  `ollama show <model>`; e.g. `llama3.2-vision`, `qwen2.5vl`, `gemma3`,
  `moondream` all support this). On a non-vision model, the tool isn't
  offered at all, so the model won't try to "imagine" an image's contents.
  Images are capped at 10MB on disk.

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
main.go                    CLI flag parsing, stdin/interactive wiring, model capability check
internal/agent/agent.go    Chat + tool-calling loop, system prompt, streaming output
internal/ollama/client.go  Minimal /api/chat, /api/tags, /api/show HTTP client
internal/tools/            The read-only tools + their JSON schemas
internal/security/path.go  Path confinement/validation helpers
internal/config/config.go  ~/.braai/settings.json persistence
```

The agent loop (`internal/agent/agent.go`) is intentionally simple:

1. Send the running message history plus the tool schemas to `/api/chat`
   (streamed, so output starts appearing as soon as the model produces it).
2. If the model's response includes tool calls, execute each one against the
   `tools.Registry` (which enforces the working-directory confinement) and
   append the results as `tool` role messages — including any attached
   images from `read_image`.
3. Repeat until the model returns a plain text answer or a configurable
   `--max-tool-calls` guardrail is hit.

`Agent.Run` returns an `agent.RunResult` (answer, reasoning, and a
`[]ToolCallRecord` of every tool call made this turn, each with its name,
arguments, and result) alongside the updated history. In text mode the
answer is streamed to `Options.Stdout` as it's produced and `RunResult` is
only used for its `History`; in `--output json` mode `Options.Stdout` is set
to `io.Discard` so nothing streams, and `main.go` marshals `RunResult`
directly once `Run` returns.

This relies on Ollama's native OpenAI-compatible tool-calling support in
`/api/chat` (message `tool_calls` / role `"tool"`). The `ollama.Message` and
`ollama.Tool` types are a thin, self-contained representation of that wire
format, so a JSON-emission fallback for models without native tool support
could be added later as an alternate implementation behind the same
`agent.Agent` interface without touching the CLI or tools layer.

Before building the tool registry, `main.go` calls `POST /api/show` once for
the selected model (`ollama.Client.ShowModel`) and uses the result for two
things:

- Whether it reports `vision` among its capabilities controls whether
  `read_image` is included in `Registry.Definitions()` at all — a model
  without vision support never even sees the tool, rather than being offered
  a tool it would call blindly.
- Its reported context window (`model_info`'s `*.context_length` field) feeds
  the rough context-usage warning described above.

Note on audio: some models report an `audio` capability via `/api/show`
(e.g. `gemma4:e4b`), but as of Ollama 0.31 the public `/api/chat` endpoint
only documents an `images` field for multimodal input — there's no equivalent
for attaching raw audio. So unlike `read_image`, there's currently no
supported way to feed audio files straight into a chat message; that's why
braai leans on pre-transcribed `.txt` files for meeting audio instead of a
`read_audio` tool. Worth revisiting if Ollama adds that.

## Testing

```sh
go test ./...
```

Unit tests cover path validation (traversal, absolute paths, symlink escape)
in `internal/security` and tool behavior (binary refusal, content/name
search, stat, directory listing, batch reads, extension/sort filters,
`read_image`'s vision-capability gating, and `search_semantic`'s ranking,
embedding cache, and error surfacing) in `internal/tools`. The
`search_semantic` tests use a small in-memory fake embedder (an `embedder`
interface satisfied by `*ollama.Client`) so they don't need a real Ollama
server or model.
