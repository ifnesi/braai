// Package commands loads user-defined prompt templates ("custom commands") from
// Markdown files and expands them into prompts. Commands are invoked in the chat
// REPL via "/cmd <name> [args...]". They are pure prompt templates: expansion
// produces text that is submitted as a normal user turn, so custom commands
// never gain any capability beyond what a typed prompt has.
package commands

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Command is a single prompt template loaded from a <name>.md file.
type Command struct {
	Name        string   // invocation name (filename without .md)
	Description string   // one-line description (optional, from frontmatter)
	Args        []string // declared argument names for usage/help (optional)
	Body        string   // template text (after any frontmatter)
	Source      string   // "global" or "project"
}

// Usage renders the declared argument names, e.g. "date project".
func (c *Command) Usage() string { return strings.Join(c.Args, " ") }

// Expand substitutes template variables in the body:
//
//	$ARGS       all args joined by a space
//	$1..$9      positional args (empty string when not provided)
//	$SELECTION  caller-supplied selection text (e.g. the last answer)
//	$$          a literal "$"
//
// Unknown "$NAME" tokens are left untouched. The result is trimmed.
func (c *Command) Expand(args []string, selection string) string {
	const sentinel = "\x00DOLLAR\x00"
	body := strings.ReplaceAll(c.Body, "$$", sentinel)
	body = strings.ReplaceAll(body, "$ARGS", strings.Join(args, " "))
	body = strings.ReplaceAll(body, "$SELECTION", selection)
	// Replace $9..$1 (high-to-low so $1 doesn't shadow $10-style tokens, though
	// only single digits are supported here).
	for i := 9; i >= 1; i-- {
		val := ""
		if i-1 < len(args) {
			val = args[i-1]
		}
		body = strings.ReplaceAll(body, "$"+strconv.Itoa(i), val)
	}
	body = strings.ReplaceAll(body, sentinel, "$")
	return strings.TrimSpace(body)
}

// Load reads *.md command files from globalDir then projectDir. A command in
// projectDir overrides a same-named command from globalDir. Missing or
// unreadable directories are ignored (not errors). Keys are command names.
func Load(globalDir, projectDir string) map[string]*Command {
	cmds := map[string]*Command{}
	loadDir(cmds, globalDir, "global")
	loadDir(cmds, projectDir, "project") // project overrides global
	return cmds
}

func loadDir(cmds map[string]*Command, dir, source string) {
	if dir == "" {
		return
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".md") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		name := strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
		if name == "" {
			continue
		}
		c := parse(name, string(data))
		c.Source = source
		cmds[name] = c
	}
}

// parse extracts optional frontmatter (between a leading "---" line and the next
// "---" line) for "description" and "args", returning the remaining body. A file
// without valid frontmatter is treated entirely as the body.
func parse(name, content string) *Command {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	c := &Command{Name: name, Body: strings.TrimSpace(content)}

	if !strings.HasPrefix(content, "---\n") {
		return c
	}
	rest := content[len("---\n"):]
	end := strings.Index(rest, "\n---")
	if end < 0 {
		return c // malformed frontmatter: whole file is the body
	}
	front := rest[:end]

	// Body begins after the rest of the closing delimiter line.
	body := ""
	if nl := strings.IndexByte(rest[end+len("\n---"):], '\n'); nl >= 0 {
		body = rest[end+len("\n---")+nl+1:]
	}
	c.Body = strings.TrimSpace(body)

	for _, line := range strings.Split(front, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		kv := strings.SplitN(line, ":", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(strings.ToLower(kv[0]))
		val := strings.TrimSpace(kv[1])
		switch key {
		case "description":
			c.Description = strings.Trim(val, `"'`)
		case "args":
			c.Args = parseArgs(val)
		}
	}
	return c
}

// parseArgs parses "args" values like "[date, project]", "date, project", or
// "date project" into ["date", "project"].
func parseArgs(val string) []string {
	val = strings.TrimSpace(val)
	val = strings.TrimPrefix(val, "[")
	val = strings.TrimSuffix(val, "]")
	fields := strings.FieldsFunc(val, func(r rune) bool { return r == ',' || r == ' ' })
	var out []string
	for _, f := range fields {
		if f = strings.Trim(f, `"' `); f != "" {
			out = append(out, f)
		}
	}
	return out
}
