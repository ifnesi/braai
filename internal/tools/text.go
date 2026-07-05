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
	// io.ReadFull fills buf (up to sniffLen) so the NUL check inspects the whole
	// sniff window; a short first Read no longer ends the sniff early. EOF (empty
	// file) and ErrUnexpectedEOF (file shorter than sniffLen) just mean n<sniffLen,
	// which is fine.
	n, err := io.ReadFull(f, buf)
	if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
		return false, err
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
