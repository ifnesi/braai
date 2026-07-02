package tools

import (
	"bytes"
	"errors"
	"io"
	"os"
)

// sniffLen is how many leading bytes we inspect to guess text vs binary.
const sniffLen = 8192

// looksLikeText applies a practical heuristic: read a leading chunk and
// reject the file as binary if it contains a NUL byte, which virtually never
// appears in legitimate text files but is common in binaries.
func looksLikeText(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()

	buf := make([]byte, sniffLen)
	n, err := f.Read(buf)
	if err != nil && n == 0 {
		// Empty file: treat as text. Other read errors bubble up.
		if errors.Is(err, io.EOF) {
			return true, nil
		}
		return false, err
	}
	return !bytes.ContainsRune(buf[:n], 0), nil
}
