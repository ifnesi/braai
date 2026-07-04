// Package crypt provides small AES-256-GCM helpers and a machine-local key
// loader, shared by braai features that encrypt data at rest (the semantic
// cache and the chat recall history). It intentionally matches the cache's
// scheme: a 32-byte key stored 0600 at ~/.braai/cache.key, and ciphertext of
// the form nonce || AES-256-GCM(ciphertext+tag).
package crypt

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"errors"
	"os"
	"path/filepath"
)

// LoadOrCreateKey returns the 32-byte key at path, generating it (0600) on first
// use. A wrong-sized existing file is treated as absent and regenerated. This is
// byte-compatible with the cache's key handling, so both share ~/.braai/cache.key.
func LoadOrCreateKey(path string) ([]byte, error) {
	if b, err := os.ReadFile(path); err == nil && len(b) == 32 {
		return b, nil
	}
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	if err := WriteFileSecure(path, key); err != nil {
		return nil, err
	}
	return key, nil
}

// WriteFileSecure writes data and forces 0600 regardless of umask.
func WriteFileSecure(path string, data []byte) error {
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return err
	}
	return os.Chmod(path, 0o600)
}

func gcm(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

// Encrypt returns nonce || AES-256-GCM(plaintext), authenticated.
func Encrypt(key, plaintext []byte) ([]byte, error) {
	g, err := gcm(key)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, g.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	// Seal appends the ciphertext to the nonce, so the result is nonce||ct.
	return g.Seal(nonce, nonce, plaintext, nil), nil
}

// Decrypt reverses Encrypt. It returns an error on a wrong key, tampering, or a
// truncated payload.
func Decrypt(key, data []byte) ([]byte, error) {
	g, err := gcm(key)
	if err != nil {
		return nil, err
	}
	ns := g.NonceSize()
	if len(data) < ns {
		return nil, errors.New("crypt: ciphertext too short")
	}
	return g.Open(nil, data[:ns], data[ns:], nil)
}
