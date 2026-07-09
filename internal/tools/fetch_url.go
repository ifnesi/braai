package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	nurl "net/url"
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

// errBlockedAddr is returned by safeDialContext when a resolved address is
// private/loopback/link-local/reserved, so fetchURL can surface it as a
// clean "HTTP 0" result instead of a raw dial error.
var errBlockedAddr = fmt.Errorf("blocked: address is private, loopback, link-local, or otherwise reserved")

// isBlockedIP reports whether ip must not be reached by fetch_url. This
// closes the SSRF hole where a model (potentially steered by content it
// already fetched, i.e. prompt injection) could otherwise reach cloud
// metadata endpoints (169.254.169.254), localhost services, or internal
// network hosts. Scheme/host allowlisting alone doesn't help here since
// attacker-controlled hostnames can resolve to any address.
func isBlockedIP(ip net.IP) bool {
	return ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsMulticast() ||
		ip.IsUnspecified()
}

// safeDialContext resolves addr and refuses to connect if any candidate
// address is blocked (see isBlockedIP), then dials the first allowed address
// directly by IP. Dialing by the resolved IP — rather than letting the
// standard dialer re-resolve the hostname — avoids a DNS-rebinding
// TOCTOU window between the check and the actual connection. Because this
// is wired in as the http.Transport's DialContext, it re-runs on every
// redirect hop (each hop opens a new connection through the same
// Transport), so a redirect to an internal address is blocked too, not just
// the initial URL.
func safeDialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}

	var ips []net.IP
	if ip := net.ParseIP(host); ip != nil {
		ips = []net.IP{ip}
	} else {
		addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			return nil, err
		}
		for _, a := range addrs {
			ips = append(ips, a.IP)
		}
	}
	if len(ips) == 0 {
		return nil, fmt.Errorf("no addresses found for %s", host)
	}
	for _, ip := range ips {
		if isBlockedIP(ip) {
			return nil, errBlockedAddr
		}
	}

	dialer := net.Dialer{}
	return dialer.DialContext(ctx, network, net.JoinHostPort(ips[0].String(), port))
}

// fetchDialContext is a package-level indirection so tests can substitute a
// plain dialer and exercise fetchURL end-to-end against an httptest.Server
// (which listens on 127.0.0.1 — a loopback address safeDialContext blocks by
// design). Production code always uses safeDialContext.
var fetchDialContext = safeDialContext

// checkRedirectHTTPSOnly is the http.Client.CheckRedirect used when
// fetch_url_https_only is enabled: it rejects any redirect hop whose target
// scheme has downgraded to plain http, and caps the redirect chain length.
// Extracted to a named function (rather than an inline closure) so it can be
// unit-tested directly against synthetic *http.Request values, without
// needing a real TLS-to-HTTP redirect over the network.
func checkRedirectHTTPSOnly(req *http.Request, via []*http.Request) error {
	if strings.ToLower(req.URL.Scheme) == "http" {
		return fmt.Errorf("redirect to plain http:// blocked by fetch_url_https_only")
	}
	if len(via) >= 10 {
		return fmt.Errorf("stopped after 10 redirects")
	}
	return nil
}

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
	// Defense in depth: Definitions() already omits fetch_url from the model's
	// tool list when disabled, but that only stops a well-behaved model from
	// choosing it — it doesn't stop Call() from dispatching a hallucinated or
	// replayed call by name. Mirrors the same check readImage() does for
	// visionCapable.
	if !r.fetchURLEnabled {
		return Result{}, fmt.Errorf("fetch_url is disabled; set fetch_url_enabled = true in braai.conf to enable it")
	}

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
		Timeout:   time.Duration(r.fetchURLTimeoutSeconds) * time.Second,
		Transport: &http.Transport{DialContext: fetchDialContext},
	}

	// Block HTTPS→HTTP downgrade redirects when https_only is enabled.
	if r.fetchURLHTTPSOnly {
		client.CheckRedirect = checkRedirectHTTPSOnly
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
	if !strings.Contains(contentType, "text/html") && !looksLikeHTML(body) {
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

// looksLikeHTML sniffs the leading bytes of body for an HTML doctype/tag.
// Used as a fallback when a server omits or mislabels Content-Type (a
// surprising number do), so those pages still get readability + Markdown
// treatment instead of being returned as raw markup.
func looksLikeHTML(body []byte) bool {
	head := body
	if len(head) > 512 {
		head = head[:512]
	}
	head = bytes.TrimSpace(head)
	lower := strings.ToLower(string(head))
	return strings.HasPrefix(lower, "<!doctype html") || strings.HasPrefix(lower, "<html")
}
