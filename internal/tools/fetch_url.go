package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	nurl "net/url"
	"net/http"
	"strings"
	"time"

	readability "codeberg.org/readeck/go-readability/v2"
	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"braai/internal/ollama"
)

// fetchMaxExtractedBytes is the hard cap applied to the extracted text
// returned to the model. Raw HTML pages (e.g. Wikipedia) can expand into
// hundreds of KiB of Markdown even after readability filtering, so we cap
// the extracted result independently of the body download cap.
const fetchMaxExtractedBytes = 32 * 1024 // 32 KiB of extracted text

func fetchURLDefinition() ollama.Tool {
	return ollama.Tool{
		Type: "function",
		Function: ollama.ToolFunction{
			Name: "fetch_url",
			Description: "Fetch a URL and return its main text content together with the HTTP status code. " +
				"HTML pages are processed through a readability filter to extract the main article body " +
				"(stripping navigation, sidebars, ads, and footers), then converted to Markdown. " +
				"Returns the status code on the first line (e.g. \"HTTP 200\") followed by the extracted text. " +
				"Network or TLS errors are returned as \"HTTP 0\" with an error description — no retries are performed; " +
				"decide based on the response whether to retry or escalate. " +
				"Only available when fetch_url_enabled = true in braai.conf. " +
				"By default only https:// URLs are accepted (fetch_url_https_only = true).",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "The URL to fetch (e.g. https://example.com/page).",
					},
				},
				"required": []string{"url"},
			},
		},
	}
}

// fetchURL performs an HTTP GET on the given URL, extracts readable text from
// the response body, and returns it as a text result. All errors (network,
// TLS, HTTP 4xx/5xx) are surfaced in the result text — not as Go errors — so
// the agent loop keeps the conversation going and the LLM can decide how to
// respond. No retries are performed.
func (r *Registry) fetchURL(ctx context.Context, args map[string]any) (Result, error) {
	rawURL, err := stringArg(args, "url")
	if err != nil {
		return Result{}, err
	}

	// Parse and validate scheme.
	parsed, err := nurl.Parse(rawURL)
	if err != nil {
		return textResult(fmt.Sprintf("HTTP 0\nError: invalid URL: %s", err.Error())), nil
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "https" && scheme != "http" {
		return textResult(fmt.Sprintf("HTTP 0\nError: unsupported URL scheme %q; only http and https are allowed", parsed.Scheme)), nil
	}
	if r.fetchURLHTTPSOnly && scheme == "http" {
		return textResult("HTTP 0\nError: plain http:// URLs are not allowed; fetch_url_https_only is enabled. " +
			"Set fetch_url_https_only = false in braai.conf to allow plain HTTP."), nil
	}

	client := &http.Client{
		Timeout: time.Duration(r.fetchURLTimeoutSeconds) * time.Second,
	}

	// Block HTTPS→HTTP downgrade redirects when https_only is enabled.
	if r.fetchURLHTTPSOnly {
		client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
			if strings.ToLower(req.URL.Scheme) == "http" {
				return fmt.Errorf("redirect to plain http:// blocked by fetch_url_https_only")
			}
			if len(via) >= 10 {
				return fmt.Errorf("stopped after 10 redirects")
			}
			return nil
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return textResult(fmt.Sprintf("HTTP 0\nError: could not build request: %s", err.Error())), nil
	}
	req.Header.Set("User-Agent", "braai/fetch_url")

	resp, err := client.Do(req)
	if err != nil {
		return textResult(fmt.Sprintf("HTTP 0\nError: %s", err.Error())), nil
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, int64(r.fetchURLMaxBytes)))
	if err != nil {
		return textResult(fmt.Sprintf("HTTP %d\nError: failed to read response body: %s", resp.StatusCode, err.Error())), nil
	}

	// Extract readable text. For HTML, run readability first (strips nav/ads/
	// sidebars/footers to just the main article body), then convert to Markdown.
	// For non-HTML content types pass the raw body through as-is.
	text := extractText(rawURL, resp.Header.Get("Content-Type"), bodyBytes)

	// Cap extracted text to avoid flooding the model context window.
	// Append a note so the model knows the content was truncated.
	if len(text) > fetchMaxExtractedBytes {
		text = text[:fetchMaxExtractedBytes] + "\n\n[content truncated]"
	}

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return textResult(fmt.Sprintf("HTTP %d\n\n%s", resp.StatusCode, text)), nil
	}
	// Non-2xx: include the status text as an error line, then the body if any.
	statusText := http.StatusText(resp.StatusCode)
	if text != "" {
		return textResult(fmt.Sprintf("HTTP %d\nError: %s\n\n%s", resp.StatusCode, statusText, text)), nil
	}
	return textResult(fmt.Sprintf("HTTP %d\nError: %s", resp.StatusCode, statusText)), nil
}

// extractText converts raw HTML bytes into clean Markdown. For HTML it first
// runs a readability filter to extract the main article content (stripping nav,
// sidebars, ads, footers), then converts the cleaned HTML to Markdown.
// Non-HTML content is returned as-is.
func extractText(pageURL, contentType string, body []byte) string {
	if !strings.Contains(contentType, "text/html") {
		return string(body)
	}

	// Attempt readability extraction to strip nav/footer/sidebar noise.
	// FromReader needs a *url.URL; parse it, ignoring errors (nil is fine).
	parsedURL, _ := nurl.Parse(pageURL)
	article, err := readability.FromReader(bytes.NewReader(body), parsedURL)
	if err == nil && article.Node != nil {
		// Render the cleaned-up article HTML back to a string, then convert
		// that to Markdown. This gives much better structure than plain text.
		var buf bytes.Buffer
		if rerr := article.RenderHTML(&buf); rerr == nil && buf.Len() > 0 {
			if md, merr := htmltomarkdown.ConvertString(buf.String()); merr == nil {
				return md
			}
		}
		// RenderHTML failed — fall back to the plain-text render.
		buf.Reset()
		if rerr := article.RenderText(&buf); rerr == nil && buf.Len() > 0 {
			return buf.String()
		}
	}

	// Readability failed entirely — convert the full page HTML to Markdown.
	if md, err := htmltomarkdown.ConvertString(string(body)); err == nil {
		return md
	}
	return string(body)
}
