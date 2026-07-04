package textextract

import (
	"testing"
)

func TestExtractAndChunk(t *testing.T) {
	// Test extraction and chunking on a simple markdown file
	text := `# Section 1

This is the first paragraph of section 1. It contains some text about authentication.

Authentication requires OAuth 2.0 or SAML. Multi-factor authentication is needed for admins.

# Section 2

This is the second section. It talks about security.

Security includes encryption with AES-256 and TLS 1.3 for data in transit.

# Section 3

The third section is about compliance.

Compliance with GDPR, HIPAA, and SOC 2 Type II is required.`

	// Test token estimation
	tokens := EstimateTokens(text)
	if tokens < 50 {
		t.Fatalf("expected at least 50 tokens, got %d", tokens)
	}

	// Test cleaning
	cleaned := CleanForLLM(text)
	if len(cleaned) == 0 {
		t.Fatal("cleaned text should not be empty")
	}

	// Test chunking with small max_tokens to force multiple chunks
	chunks := ChunkText(cleaned, 100, "test.md")
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks with 100-token limit, got %d", len(chunks))
	}

	// Verify chunk metadata
	for i, chunk := range chunks {
		if chunk.Index != i+1 {
			t.Fatalf("chunk %d has wrong Index: %d", i, chunk.Index)
		}
		if chunk.Total != len(chunks) {
			t.Fatalf("chunk %d has wrong Total: %d (should be %d)", i, chunk.Total, len(chunks))
		}
		if chunk.Source != "test.md" {
			t.Fatalf("chunk %d has wrong Source: %s", i, chunk.Source)
		}
		if chunk.Tokens < 1 {
			t.Fatalf("chunk %d has invalid token count: %d", i, chunk.Tokens)
		}
	}

	// Test manifest generation
	manifest := BuildManifest(chunks)
	if len(manifest) != len(chunks) {
		t.Fatalf("manifest should have %d entries, got %d", len(chunks), len(manifest))
	}
	for i, entry := range manifest {
		if entry.Index != i+1 {
			t.Fatalf("manifest entry %d has wrong Index: %d", i, entry.Index)
		}
		if entry.Summary == "" {
			t.Fatalf("manifest entry %d has empty summary", i)
		}
	}
}

func TestNormalizeWhitespace(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello  world", "hello  world"},     // preserves internal spaces
		{"hello\n\nworld", "hello\n\nworld"}, // consecutive blanks are consolidated
		{"  hello  ", "hello"},
		{"hello\n\n\n\nworld", "hello\n\nworld"},
	}

	for _, test := range tests {
		result := NormalizeWhitespace(test.input)
		if result != test.expected {
			t.Errorf("NormalizeWhitespace(%q) = %q, expected %q", test.input, result, test.expected)
		}
	}
}

func TestEstimateTokens(t *testing.T) {
	text := "This is a test sentence."
	tokens := EstimateTokens(text)
	if tokens < 1 {
		t.Fatalf("EstimateTokens should return at least 1 for non-empty text, got %d", tokens)
	}

	empty := EstimateTokens("")
	if empty != 0 {
		t.Fatalf("EstimateTokens should return 0 for empty string, got %d", empty)
	}

	// EstimateTokens counts runes, not words; whitespace-only has ~2-3 tokens
	whitespace := EstimateTokens("   \n  \t  ")
	if whitespace > 5 {
		t.Fatalf("EstimateTokens for whitespace should be very low, got %d", whitespace)
	}
}
