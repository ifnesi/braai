// Package history persists the interactive chat's up/down-arrow recall entries
// encrypted at rest, so ~/.braai/chat_history never contains plaintext prompts.
// It reuses the shared crypt package (and thus the same ~/.braai/cache.key).
package history

import (
	"os"
	"strings"

	"braai/internal/crypt"
)

// Store holds the recall history in memory and mirrors it to an encrypted file.
// Not safe for concurrent access; assumes single-threaded use within a session.
type Store struct {
	path  string // empty => persistence disabled (in-memory only)
	key   []byte
	limit int
	lines []string
}

// Open loads (and decrypts) the history at path, creating/loading the key at
// keyPath. If the file is missing it starts empty. If it exists but cannot be
// decrypted (legacy plaintext from an older braai, a changed key, or
// corruption), its contents are discarded AND it is immediately overwritten
// with an empty encrypted blob, so no plaintext is left on disk. An empty path
// disables persistence entirely (in-memory recall only).
func Open(path, keyPath string, limit int) (*Store, error) {
	if path == "" {
		return &Store{limit: limit}, nil
	}
	key, err := crypt.LoadOrCreateKey(keyPath)
	if err != nil {
		return nil, err
	}
	s := &Store{path: path, key: key, limit: limit}

	data, err := os.ReadFile(path)
	if err != nil || len(data) == 0 {
		return s, nil // missing/empty/unreadable: start empty (best-effort)
	}
	plain, derr := crypt.Decrypt(key, data)
	if derr != nil {
		// Do not expose whatever is there; scrub it to an empty encrypted file.
		_ = s.persist()
		return s, nil
	}
	for _, ln := range strings.Split(string(plain), "\n") {
		if ln != "" {
			s.lines = append(s.lines, ln)
		}
	}
	return s, nil
}

// Lines returns the recall entries oldest-first (for seeding readline).
func (s *Store) Lines() []string {
	if s == nil {
		return nil
	}
	return s.lines
}

// Add appends a new entry (skipping blanks and consecutive duplicates), trims to
// the limit, and re-persists the encrypted file.
func (s *Store) Add(line string) error {
	if s == nil || s.path == "" {
		return nil
	}
	line = strings.TrimRight(line, "\r\n")
	if line == "" {
		return nil
	}
	if n := len(s.lines); n > 0 && s.lines[n-1] == line {
		return nil
	}
	s.lines = append(s.lines, line)
	if s.limit > 0 && len(s.lines) > s.limit {
		s.lines = s.lines[len(s.lines)-s.limit:]
	}
	return s.persist()
}

// Clear erases all entries in memory and on disk (leaving an empty encrypted file).
func (s *Store) Clear() error {
	if s == nil {
		return nil
	}
	s.lines = nil
	return s.persist()
}

func (s *Store) persist() error {
	if s.path == "" {
		return nil
	}
	content := strings.Join(s.lines, "\n")
	blob, err := crypt.Encrypt(s.key, []byte(content))
	if err != nil {
		return err
	}
	return crypt.WriteFileSecure(s.path, blob)
}
