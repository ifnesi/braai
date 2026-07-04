# braai

`braai` is a small, read-only AI agent CLI that answers questions about a local
working directory. It uses a local [Ollama](https://ollama.com) server for
reasoning and lets the model call a fixed set of read-only filesystem tools
(list, read, batch-read, search by name/content, semantic search across the
whole tree, read/extract/search documents, stat, and — on vision-capable
models — read/OCR images) to gather evidence before answering. It never writes,
deletes, or executes anything on your filesystem.

Semantic search runs on a **fast, in-process static embedding model** that braai
downloads once from Hugging Face and runs itself — no Ollama embedding model and
no server round-trip are needed for embeddings. Ollama is used only for the chat
model.

It works well as a local research assistant over a folder of meeting notes and
audio transcripts, e.g. `braai --working-dir ~/notes/2026-Q3 "summarize this
week's meetings and flag any action items"` — see
[Read-only toolset](#read-only-toolset) for the tools it uses to do that.

![Image](img/braai.png)

## Requirements

- Go 1.21+ (the module currently pins `go 1.25`; lower the `go` directive in
  `go.mod` if you need to build with an older toolchain).
- A running local Ollama server (`ollama serve`) with at least one model
  pulled (e.g. `ollama pull llama3.1`). The model should support Ollama's
  native tool calling for best results (check with `ollama show <model>` —
  look for `tools` under capabilities).
- Network access **on first use of semantic search only**, to download the
  static embedding model from Hugging Face into `~/.braai/models/`. After that,
  embeddings work fully offline.
- Optional: `pdftotext` (from `poppler-utils`) on your `PATH` for higher-quality
  PDF extraction. braai falls back to a pure-Go PDF reader when it's absent.

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

# Reasoning/thinking is streamed before the answer by default; opt out with --hide-reasoning
./braai --model llama3.1 --hide-reasoning "why does main.go split runOnce and runChat?"

# Get a single structured JSON object instead of streamed text, e.g. for scripting
./braai --model llama3.1 --output json "list the files in internal/tools" | jq .
```

By default (`--output text`), answers are streamed to stdout as the model
produces them, rather than printed only once the full response is ready. The
model's reasoning/thinking trace (on models that support it) is streamed
before the answer by default too — pass `--hide-reasoning` to suppress it.
In a real terminal, the reasoning trace is dimmed so it's visually distinct
from the final answer; the dimming is skipped automatically when stdout is
piped or redirected, so redirected output stays free of ANSI escape codes.

With `--output json`, streaming is disabled and braai instead prints one
JSON object per answer once it's complete:
`{"answer": "...", "reasoning": "...", "tool_calls": [{"name": "...", "arguments": {...}, "result": "..."}]}`.
`reasoning` is omitted when `--hide-reasoning` is set, and `tool_calls` is
omitted if the model answered without using any tools. This works in both
one-shot and interactive chat mode (one JSON object per turn).

The interactive chat uses a `>>>` prompt and supports standard readline-style
line editing: left/right arrows to move within the line, Ctrl-A/Ctrl-E to
jump to the start/end, Ctrl-C to clear the current input (shows a hint to exit
via Ctrl + d or `/bye`), and up/down arrows to recall history from previous
sessions. Chat history is persisted to `~/.braai/chat_history` **encrypted at rest**
with AES-256-GCM (using the machine-local key at `~/.braai/cache.key`), so no
plaintext prompts are written to disk. The history limit is configured in
`~/.braai/braai.conf` via `history_limit` (default: 100 entries).

When the model produces reasoning/thinking (on models that support it), it's
shown with bold **Thinking...** and **...done thinking.** markers so it's
visually distinct from the final answer; pass `--hide-reasoning` to suppress it.

A few slash-commands are available inside the chat:

| Command | Effect |
|---|---|
| `/help` | List available commands |
| `/clear` | Reset the conversation history and clear the visible screen (start fresh without restarting) |
| `/copy` | Copy the last answer to clipboard |
| `/bye` | Exit the chat (same as `exit` or `quit`; Ctrl + d, also works) |
| `/forget-history` | Erase `~/.braai/chat_history` (the encrypted up/down arrow recall history, separate from the conversation itself) |
| `/tools` | List the tools currently available to the model (name + description) |
| `/tools full` | Same as `/tools`, but also shows each tool's arguments (type, whether required, and description) |
| `/cache` | Show semantic-search cache stats for the current directory (files, chunks, size on disk) |
| `/cache clear` | Delete the semantic-search cache for the current directory |
| `/model` | Show the current model and list every model available on the Ollama server |
| `/model <name>` | Switch to a different model and save it as the default (persisted to `~/.braai/braai.conf`) |
| `/save <file>` | Save the visible conversation (your messages + braai's answers) as Markdown |
| `/cmd` | List all available custom prompt-template commands |
| `/cmd <name> [args...]` | Expand and run a custom prompt template |

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
| `--model` | first available | Ollama chat model to use |
| `--embed-model` | `minishlab/potion-retrieval-32M` | Hugging Face repo of the static embedding model used for semantic search (downloaded and run in-process) |
| `--working-dir` | `.` | Root directory the agent may inspect |
| `--ollama-host` | `http://localhost:11434` | Ollama server base URL |
| `--prompt` | — | Run one prompt non-interactively (trailing args work the same way) |
| `--verbose` | `false` | Print tool calls and intermediate steps to stderr |
| `--hide-reasoning` | `false` | Don't stream the model's reasoning/thinking trace (shown by default, on models that support it) before its answer |
| `--max-tool-calls` | `100` | Max tool calls allowed per request before aborting |
| `--max-read-bytes` | `-1` (no limit) | Max bytes `read_file` will return |
| `--version` | — | Print the braai version and exit |
| `--output` | `text` | `text` streams the answer as produced; `json` buffers and prints one JSON object per answer (`answer`, `reasoning`, `tool_calls`) |

### Configuration file

`braai` persists your last-used Ollama host, chat model, embedding model, and
other settings to `~/.braai/braai.conf` (key=value format with comments) so
subsequent runs can reuse them as defaults. Command-line flags always take
precedence over this file, and values you set by hand in the file are preserved
(they aren't clobbered by runtime defaults).

Example `~/.braai/braai.conf`:

```conf
# braai configuration
# Key=value format. Lines starting with # are comments.

# Ollama API host
ollama_host=http://localhost:11434

# Default chat model
model=gemma4:12b-mlx

# Default embedding model (Hugging Face repo)
embed_model=minishlab/potion-retrieval-32M

# Maximum tool calls per request
max_tool_calls=100

# Maximum number of recall history entries
history_limit=100

# Cache extracted text from documents (true/false)
cache_extracted_text=true

# Cache compression method (flate or none)
cache_compression=flate

# Encrypt cache at rest (true/false)
cache_encryption=true

# Total cache size in bytes (0 = 1 GiB default)
cache_max_bytes=0
```

Core settings:

- `ollama_host` — URL of your Ollama server (default: `http://localhost:11434`)
- `model` — Default chat model (auto-detected from first available if omitted)
- `embed_model` — Hugging Face repo of the static embedding model used for
  semantic search (default: `minishlab/potion-retrieval-32M`). This is **not**
  an Ollama model — it's a model2vec static-embedding repo that braai downloads
  to `~/.braai/models/` and runs in-process. Other options include
  `minishlab/potion-multilingual-128M` (multilingual). Changing this transparently
  rebuilds the semantic cache, since embeddings from different models aren't
  comparable.
- `max_tool_calls` — Max tool invocations per response before aborting (default: `100`)
- `history_limit` — Max number of chat history entries kept for up/down arrow recall
  (default: `100`)

Semantic-cache settings (all optional; secure defaults apply when omitted):

- `cache_extracted_text` — Persist extracted document text to disk so `get_chunk`
  is instant and doesn't re-extract (default: `true`). Set to `false` for a
  privacy-first mode: only embeddings/metadata are cached, and document text is
  re-extracted on demand — no document text is ever written to disk.
- `cache_compression` — `flate` (default) or `none`. Compresses cached text blobs.
- `cache_encryption` — Encrypt cached text blobs at rest with AES-256-GCM
  (default: `true`). The key is a machine-local file at `~/.braai/cache.key`
  (see [Security model](#security-model)).
- `cache_max_bytes` — Total on-disk budget for cached blobs before least-recently-used
  eviction kicks in (default: `0`, i.e. 1 GiB). Set to `0` to use the default; any
  positive value caps the cache size.

### Custom prompt-template commands

Create reusable prompt templates as Markdown files to extend braai's chat interface with custom commands. Templates are pure prompts — they expand and submit as normal turns, with no new security surface.

**Locations:**
- Global: `~/.braai/commands/*.md`
- Per-project: `<working-dir>/.braai/commands/*.md` (overrides global)

**Template variables:**
- `$ARGS` — all arguments joined by spaces
- `$1`–`$9` — positional arguments (empty string if not supplied)
- `$SELECTION` — text of the most recent assistant answer (empty if none)
- `$$` — a literal `$`

**Frontmatter (optional):**
Markdown files can start with YAML-ish frontmatter between `---` delimiters to set a description and declare argument names (for listing and documentation):

```markdown
---
description: Draft a standup from recent notes
args: [date, project]
---
Read my notes and produce a standup for $1 on project $2.
Focus on: $ARGS
```

**Usage:**

```
>>> /cmd                          # List available commands (global + per-project)
>>> /cmd standup 2026-07-04       # Expand template with args, show dimmed, run as turn
>>> /cmd summary                  # Use $SELECTION to summarize the last answer
>>> /cmd nope                     # "unknown command" error
>>> [up-arrow]                    # Recalls "/cmd standup ..." (not the expanded text)
```

**Examples:**

File: `~/.braai/commands/standup.md`
```markdown
---
description: Draft a standup from recent notes
args: [date]
---
Read my notes in this directory and draft a concise standup update for $1.
List: what I did, what's next, and any blockers.
```

File: `~/.braai/commands/tldr.md`
```markdown
---
description: Summarize the previous answer in 3 bullets
---
Summarize the following in exactly three bullet points:

$SELECTION
```

File: `.braai/commands/review.md` (project-level, overrides global)
```markdown
---
description: Review code in the specified file
args: [filepath]
---
Review the code in $1 for correctness, performance, and style.
Point out any potential issues or improvements.
```

Invocation:
```
>>> /cmd standup 2026-07-04      # Expands and runs
>>> /cmd tldr                     # Summarizes previous answer
>>> /cmd review src/main.go       # Project-level command
```

## Read-only toolset

The agent can only call the tools below, all confined to `--working-dir`. Use
`/tools full` inside the chat to see each tool's exact arguments.

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
- **search_semantic** — search the **entire working directory by meaning** and
  return the most relevant *passages* (not just whole files), each with a file
  path and a `chunk_index`. The model then calls `get_chunk(path, chunk_index)`
  to read a matching passage in full. Every eligible file is extracted, chunked,
  and embedded with the in-process static model; chunks are ranked by cosine
  similarity to the query. Results and embeddings are persisted in an on-disk
  cache (see [Semantic search & caching](#semantic-search--caching)), so repeated
  searches — and searches in later sessions — are near-instant for unchanged
  files. Prefer `search_content` for known exact substrings.
- **read_document** — extract and optionally chunk text from documents (PDF,
  Word, Excel, PowerPoint, HTML, CSV, JSON, RTF, plaintext, etc.). If text
  is small (≤ 2000 tokens), returns it directly; if larger, returns a
  manifest of chunks that can be fetched individually with `get_chunk`.
  Supports `clean=true` to remove headers/footers/page numbers (default), or
  `clean=false` for raw text.
- **search_document** — semantically search *within a single document* to find
  relevant chunks (e.g., "find the authentication requirements chapter").
  Returns the top matching chunks ranked by semantic similarity, using the
  same in-process static embeddings as `search_semantic` but scoped to one file.
  Useful for large documents like manuals or specs.
- **get_chunk** — fetch the full text of a specific chunk after reading or
  searching a document (or after `search_semantic`). Call `read_document`,
  `search_document`, or `search_semantic` first to get chunk indices, then use
  this to retrieve a chunk's full text by its 1-indexed number. When the cache
  has the document's text, this is served from disk without re-extraction.
- **find_all_files** — recursively list all files under a directory. Returns
  a compact JSON list of file paths. Useful for exploring large directory
  structures and discovering files without iterating with `list_dir`.
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

There are no write, delete, rename, mkdir, chmod, or shell-execution tools
exposed to the model — this is intentional and hardcoded, not something that
can be enabled.

`.git`, `node_modules`, `vendor`, `.idea`, and `.DS_Store` are skipped by
directory listing, name search, content search, and semantic search for
usability (see `internal/tools/tools.go`).

## Semantic search & caching

Semantic search is designed to be fast, reliable, and privacy-conscious.

**In-process embeddings.** `search_semantic` and `search_document` use a
[model2vec](https://github.com/MinishLab/model2vec) static embedding model that
braai loads and runs itself. Static embeddings are a token→vector lookup plus
mean-pool plus L2-normalize — no neural forward pass, no server, microseconds
per chunk. The model files (`tokenizer.json`, `model.safetensors`, `config.json`)
are downloaded once from Hugging Face into `~/.braai/models/<repo>/` and reused
thereafter. Ollama is not involved in embeddings at all, so semantic search
keeps working even with Ollama stopped.

**Passage-level results.** Rather than ranking whole files, `search_semantic`
chunks each file and ranks chunks across the whole tree, returning `path` +
`chunk_index` + a similarity score + a short excerpt. This tells the model both
*which file* and *where in it* the match is; it then fetches the full passage
with `get_chunk`.

**Persistent, compressed, encrypted cache.** Embeddings and chunk metadata are
stored per project directory under `~/.braai/cache/`, keyed by each file's
modification time and size (and by the embedding model). Unchanged files are
never re-embedded — across runs, not just within a session — so repeat searches
are near-instant. Extracted document text is stored in per-file blobs that are
compressed (flate) and encrypted (AES-256-GCM) at rest by default. Only chunk
metadata and embedding vectors live in memory; chunk text is read (and decrypted)
from disk on demand, a few chunks at a time.

**Invalidation and footprint.** A cache entry is rebuilt automatically when its
file changes (mtime/size), when the embedding model changes, or when the on-disk
format version changes; deleting `~/.braai/cache/` by hand is always safe. When
total blob size exceeds `cache_max_bytes` (default 1 GiB), least-recently-used
entries are evicted. Use `/cache` to see stats and `/cache clear` to wipe the
current directory's cache.

**Privacy switches.** Set `cache_extracted_text: false` to keep document text
off disk entirely (embeddings still persist, but they can't be reversed into
text). Encryption is on by default; see the security notes below for the key's
threat model.

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

No tool ever opens a file for writing, and no command-execution tool is
exposed to the model, so there is no path by which the agent can modify your
filesystem. (braai itself may invoke two local helper binaries on paths it has
already validated — `pdftotext` for higher-quality PDF extraction when it's
installed, and a clipboard utility such as `pbcopy`/`xclip`/`xsel` for the
`/copy` command. Neither is controllable by the model.)

**Cache and history files.** The semantic cache and chat history can contain
readable text from your documents or chat prompts, so braai protects both:

- `~/.braai/cache/` and `~/.braai/models/` are created owner-only (`0700`).
- The cache index, all cache blobs, chat history file, and the encryption key
  (`~/.braai/cache.key`) are written owner-only (`0600`).
- Both cached document text and chat history are compressed (cache only) and
  AES-256-GCM-encrypted at rest by default, using the same machine-local key.
  This protects both if they're copied off the machine (e.g. into backups or a
  synced folder). Encryption does **not** protect against someone who already
  has full read access to your home directory (they can read the key too). For
  stronger guarantees with the cache, set `cache_extracted_text: false` so no
  document text is ever written to disk (chat history remains encrypted).

## Architecture

```
main.go                       CLI flag parsing, stdin/interactive wiring, model capability check
internal/agent/agent.go       Chat + tool-calling loop, system prompt, streaming output
internal/ollama/client.go     Minimal /api/chat, /api/tags, /api/show HTTP client
internal/tools/               The read-only tools + their JSON schemas
internal/staticembed/         In-process model2vec static embeddings (tokenizer, safetensors, HF download)
internal/cache/               Persistent, compressed, encrypted semantic-search cache
internal/commands/            Custom prompt-template command loader + variable expansion
internal/textextract/         Document extraction + chunking (PDF, Office, HTML, CSV, ...)
internal/security/path.go     Path confinement/validation helpers
internal/config/config.go     ~/.braai/braai.conf persistence + cache/model/commands dirs
internal/terminal/            TTY/color detection and styling
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

Embeddings are decoupled from Ollama via a small interface in
`internal/tools` (`Embedder`, satisfied by `*staticembed.Model`), so the
embedding backend can be swapped without touching the tools or the cache.

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
`search_semantic` tests use a small in-memory fake embedder (an `Embedder`
interface also satisfied by `*staticembed.Model`) so they don't need a real
embedding model or Ollama server.
