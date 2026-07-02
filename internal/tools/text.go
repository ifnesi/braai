package tools

import (
	"bytes"
	"errors"
	"io"
	"os"
)

// sniffLen is how many leading bytes we inspect to guess text vs binary.
const sniffLen = 8192

// looksLikeTextBytes applies the "no NUL byte in a leading chunk" heuristic
// to data already read into memory (e.g. by os.ReadFile), so callers that
// already have the bytes don't need to open the file a second time just to
// sniff it.
func looksLikeTextBytes(data []byte) bool {
	if len(data) > sniffLen {
		data = data[:sniffLen]
	}
	return !bytes.ContainsRune(data, 0)
}

// looksLikeTextFile applies the same heuristic to an already-open file,
// leaving it positioned back at the start so the caller can read it again
// without a second open() syscall.
func looksLikeTextFile(f *os.File) (bool, error) {
	buf := make([]byte, sniffLen)
	// io.ReadAtLeast (rather than a single Read) guards against the
	// documented possibility of Read returning fewer bytes than requested
	// even when more are available; ErrUnexpectedEOF/EOF just mean the file
	// is shorter than sniffLen, which is fine for sniffing.
	n, err := io.ReadAtLeast(f, buf, 1)
	if err != nil {
		if errors.Is(err, io.EOF) {
			return true, nil // empty file: treat as text
		}
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			return false, err
		}
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return false, err
	}
	return !bytes.ContainsRune(buf[:n], 0), nil
}

// looksLikeText opens path, applies the same heuristic, and closes it
// afterward. Prefer looksLikeTextFile or looksLikeTextBytes when the caller
// is going to read the file's contents right after, to avoid opening it
// twice.
func looksLikeText(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	return looksLikeTextFile(f)
}
