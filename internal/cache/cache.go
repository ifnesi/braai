// Package cache provides a persistent, compressed, encrypted on-disk cache for
// braai's semantic search. It stores, per project directory, an index of file
// entries (chunk metadata + normalized embeddings) plus one blob per file
// holding the chunk-delimited extracted text.
//
// Security posture:
//   - The cache directory tree is created 0700 (owner-only).
//   - The index file, all blob files, and the encryption key are written 0600.
//   - Extracted document text is compressed (flate) and encrypted (AES-256-GCM)
//     at rest by default, using a machine-local key at ~/.braai/cache.key.
//
// In-memory footprint is bounded: only chunk metadata and embedding vectors are
// held in RAM. Chunk text lives only on disk (in the blob) and is read+decrypted
// on demand a few chunks at a time via GetChunk.
package cache

import (
	"bytes"
	"compress/flate"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"braai/internal/crypt"
)

// schemaVersion is bumped whenever the on-disk index/blob layout or the chunking
// parameters change, so stale caches are transparently discarded and rebuilt.
const schemaVersion = 1

// entryOverheadBytes is a nominal cost charged per cached file entry (on top of
// its blob size) so that eviction still bounds the cache when text blobs are
// disabled (paranoid mode), where every BlobSize is 0. It also roughly accounts
// for the in-index embedding vectors, which are not otherwise counted.
const entryOverheadBytes int64 = 4096

// Blob/index format flags (stored in the first byte of every sealed payload).
const (
	flagCompressed byte = 1 << 0
	flagEncrypted  byte = 1 << 1
)

var errCorrupt = errors.New("cache: corrupt payload")

// Options configures cache behavior. Zero values are safe: an empty Compression
// means "none", Encrypt=false disables encryption, MaxBytes<=0 disables eviction.
type Options struct {
	CacheText   bool   // persist extracted text blobs (false = paranoid mode: re-extract on demand)
	Compression string // "flate" or "none"
	Encrypt     bool   // AES-256-GCM at rest
	MaxBytes    int64  // total blob budget before LRU eviction; <=0 = unbounded
}

// ChunkMeta is the per-chunk record kept in the index (and in RAM). Embedding is
// stored L2-normalized so cosine similarity is a plain dot product.
type ChunkMeta struct {
	Index     int       `json:"index"` // 1-based, matches textextract.Chunk.Index
	Section   string    `json:"section,omitempty"`
	Tokens    int       `json:"tokens"`
	Excerpt   string    `json:"excerpt"`
	ByteStart int       `json:"byte_start"` // offset into the (decrypted, decompressed) blob
	ByteEnd   int       `json:"byte_end"`
	Embedding []float32 `json:"embedding"`
}

// FileEntry is the per-file record: identity fingerprint + chunks (+ optional blob).
type FileEntry struct {
	RelPath    string      `json:"rel_path"`
	ModTimeNS  int64       `json:"mtime_ns"`
	Size       int64       `json:"size"`
	BlobName   string      `json:"blob_name,omitempty"`
	BlobSize   int64       `json:"blob_size,omitempty"`
	LastAccess int64       `json:"last_access"`
	Chunks     []ChunkMeta `json:"chunks"`
}

// CachedChunk is the full-text result returned by GetChunk.
type CachedChunk struct {
	Index   int
	Total   int
	Section string
	Tokens  int
	Text    string
}

type index struct {
	Schema   int                   `json:"schema"`
	ModelTag string                `json:"model_tag"`
	Files    map[string]*FileEntry `json:"files"`
}

func newIndex(modelTag string) *index {
	return &index{Schema: schemaVersion, ModelTag: modelTag, Files: make(map[string]*FileEntry)}
}

// Cache is safe for concurrent use.
type Cache struct {
	mu        sync.Mutex
	dir       string // per-project cache directory
	indexPath string
	key       []byte // 32 bytes when encryption enabled, else nil
	opts      Options
	modelTag  string
	idx       *index
	dirty     bool
}

// Open initializes (or loads) the cache for projectRoot under baseCacheDir,
// keyed to modelTag (the embedding model). If the persisted model tag or schema
// no longer matches, the project's blobs are wiped and a fresh index is started.
func Open(baseCacheDir, keyPath, projectRoot, modelTag string, opts Options) (*Cache, error) {
	if err := mkdirSecure(baseCacheDir); err != nil {
		return nil, fmt.Errorf("cache dir: %w", err)
	}
	projDir := filepath.Join(baseCacheDir, hashName(projectRoot))
	if err := mkdirSecure(projDir); err != nil {
		return nil, fmt.Errorf("project cache dir: %w", err)
	}

	var key []byte
	if opts.Encrypt {
		k, err := crypt.LoadOrCreateKey(keyPath)
		if err != nil {
			return nil, fmt.Errorf("cache key: %w", err)
		}
		key = k
	}

	c := &Cache{
		dir:       projDir,
		indexPath: filepath.Join(projDir, "index"),
		key:       key,
		opts:      opts,
		modelTag:  modelTag,
	}
	c.load()
	return c, nil
}

// Get returns the fresh cached entry for relPath, or (nil,false) if absent or
// stale (mtime/size changed). Freshness is validated by fingerprint.
func (c *Cache) Get(relPath string, modTimeNS, size int64) (*FileEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e := c.idx.Files[relPath]
	if e == nil || e.ModTimeNS != modTimeNS || e.Size != size {
		return nil, false
	}
	e.LastAccess = time.Now().Unix()
	c.dirty = true
	return e, true
}

// Put stores entry and, when CacheText is enabled, writes a compressed+encrypted
// blob built from chunkTexts. It fills each chunk's ByteStart/ByteEnd (exact, by
// construction) so GetChunk can slice the blob without any fuzzy matching.
// chunkTexts must align 1:1 with entry.Chunks.
func (c *Cache) Put(entry *FileEntry, chunkTexts []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry.LastAccess = time.Now().Unix()

	if c.opts.CacheText && len(chunkTexts) == len(entry.Chunks) && len(chunkTexts) > 0 {
		var buf bytes.Buffer
		for i, t := range chunkTexts {
			start := buf.Len()
			buf.WriteString(t)
			entry.Chunks[i].ByteStart = start
			entry.Chunks[i].ByteEnd = buf.Len()
			buf.WriteByte('\n') // separator, not part of any chunk range
		}
		sealed, err := c.seal(buf.Bytes())
		if err == nil {
			name := hashName(entry.RelPath) + ".blob"
			if werr := crypt.WriteFileSecure(filepath.Join(c.dir, name), sealed); werr == nil {
				entry.BlobName = name
				entry.BlobSize = int64(len(sealed))
			}
		}
	}

	c.idx.Files[entry.RelPath] = entry
	c.dirty = true
	c.evictLocked()
	return nil
}

// GetChunk returns the full text (and metadata) of a single chunk from the blob,
// or (nil,false) on miss/stale/no-text/corruption. Callers fall back to
// re-extraction on false, so this is always safe.
func (c *Cache) GetChunk(relPath string, modTimeNS, size int64, chunkIndex int) (*CachedChunk, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	e := c.idx.Files[relPath]
	if e == nil || e.ModTimeNS != modTimeNS || e.Size != size || e.BlobName == "" {
		return nil, false
	}
	var meta *ChunkMeta
	for i := range e.Chunks {
		if e.Chunks[i].Index == chunkIndex {
			meta = &e.Chunks[i]
			break
		}
	}
	if meta == nil {
		return nil, false
	}
	blob, err := os.ReadFile(filepath.Join(c.dir, e.BlobName))
	if err != nil {
		return nil, false
	}
	plain, err := c.open(blob)
	if err != nil || meta.ByteStart < 0 || meta.ByteEnd > len(plain) || meta.ByteStart > meta.ByteEnd {
		return nil, false
	}
	e.LastAccess = time.Now().Unix()
	c.dirty = true
	return &CachedChunk{
		Index:   meta.Index,
		Total:   len(e.Chunks),
		Section: meta.Section,
		Tokens:  meta.Tokens,
		Text:    string(plain[meta.ByteStart:meta.ByteEnd]),
	}, true
}

// Flush persists the index to disk (atomic rename). No-op when nothing changed.
func (c *Cache) Flush() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.flushLocked()
}

// Status reports cache size for the current project.
func (c *Cache) Status() (files, chunks int, bytes int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.idx.Files {
		files++
		chunks += len(e.Chunks)
		bytes += e.BlobSize
	}
	return
}

// Clear wipes this project's blobs and index from disk.
func (c *Cache) Clear() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.wipeBlobsLocked()
	c.idx = newIndex(c.modelTag)
	c.dirty = true
	return c.flushLocked()
}

// --- internals ---

func (c *Cache) flushLocked() error {
	if !c.dirty {
		return nil
	}
	raw, err := json.Marshal(c.idx)
	if err != nil {
		return err
	}
	sealed, err := c.seal(raw)
	if err != nil {
		return err
	}
	tmp := c.indexPath + ".tmp"
	if err := crypt.WriteFileSecure(tmp, sealed); err != nil {
		return err
	}
	if err := os.Rename(tmp, c.indexPath); err != nil {
		os.Remove(tmp)
		return err
	}
	c.dirty = false
	return nil
}

func (c *Cache) load() {
	b, err := os.ReadFile(c.indexPath)
	if err != nil {
		c.idx = newIndex(c.modelTag)
		return
	}
	plain, err := c.open(b)
	if err != nil {
		c.idx = newIndex(c.modelTag)
		return
	}
	var idx index
	if json.Unmarshal(plain, &idx) != nil || idx.Files == nil {
		c.idx = newIndex(c.modelTag)
		return
	}
	if idx.Schema != schemaVersion || idx.ModelTag != c.modelTag {
		// Model or layout changed: old vectors/offsets are incomparable/invalid.
		c.wipeBlobsLocked()
		c.idx = newIndex(c.modelTag)
		c.dirty = true
		return
	}
	c.idx = &idx
}

func (c *Cache) evictLocked() {
	if c.opts.MaxBytes <= 0 {
		return
	}
	cost := func(e *FileEntry) int64 { return e.BlobSize + entryOverheadBytes }

	var total int64
	for _, e := range c.idx.Files {
		total += cost(e)
	}
	if total <= c.opts.MaxBytes {
		return
	}
	type kv struct {
		path string
		e    *FileEntry
	}
	list := make([]kv, 0, len(c.idx.Files))
	for p, e := range c.idx.Files {
		list = append(list, kv{p, e})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].e.LastAccess < list[j].e.LastAccess })
	for _, item := range list {
		if total <= c.opts.MaxBytes {
			break
		}
		if item.e.BlobName != "" {
			os.Remove(filepath.Join(c.dir, item.e.BlobName))
		}
		total -= cost(item.e)
		delete(c.idx.Files, item.path)
	}
	c.dirty = true
}

func (c *Cache) wipeBlobsLocked() {
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		return
	}
	for _, ent := range entries {
		if filepath.Ext(ent.Name()) == ".blob" {
			os.Remove(filepath.Join(c.dir, ent.Name()))
		}
	}
}

// seal compresses (optional) then encrypts (optional), prepending a flags byte.
func (c *Cache) seal(plain []byte) ([]byte, error) {
	var flags byte
	data := plain

	if c.opts.Compression == "flate" {
		var buf bytes.Buffer
		w, err := flate.NewWriter(&buf, flate.BestSpeed)
		if err != nil {
			return nil, err
		}
		if _, err := w.Write(data); err != nil {
			return nil, err
		}
		if err := w.Close(); err != nil {
			return nil, err
		}
		data = buf.Bytes()
		flags |= flagCompressed
	}

	if c.opts.Encrypt && c.key != nil {
		gcm, err := newGCM(c.key)
		if err != nil {
			return nil, err
		}
		nonce := make([]byte, gcm.NonceSize())
		if _, err := rand.Read(nonce); err != nil {
			return nil, err
		}
		ct := gcm.Seal(nil, nonce, data, nil)
		data = append(nonce, ct...)
		flags |= flagEncrypted
	}

	return append([]byte{flags}, data...), nil
}

// open reverses seal, using the self-describing flags byte (so blobs written
// under different settings still decode, as long as the key is available).
func (c *Cache) open(blob []byte) ([]byte, error) {
	if len(blob) < 1 {
		return nil, errCorrupt
	}
	flags := blob[0]
	data := blob[1:]

	if flags&flagEncrypted != 0 {
		if c.key == nil {
			return nil, errors.New("cache: encrypted payload but no key")
		}
		gcm, err := newGCM(c.key)
		if err != nil {
			return nil, err
		}
		ns := gcm.NonceSize()
		if len(data) < ns {
			return nil, errCorrupt
		}
		pt, err := gcm.Open(nil, data[:ns], data[ns:], nil)
		if err != nil {
			return nil, err
		}
		data = pt
	}

	if flags&flagCompressed != 0 {
		r := flate.NewReader(bytes.NewReader(data))
		out, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			return nil, err
		}
		data = out
	}

	return data, nil
}

func newGCM(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

func hashName(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// mkdirSecure creates dir (and parents) and forces 0700 regardless of umask.
func mkdirSecure(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	return os.Chmod(dir, 0o700)
}
