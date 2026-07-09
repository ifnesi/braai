package tools

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"braai/internal/security"
)

// withPlainDialer temporarily swaps fetchDialContext for a plain dialer so
// tests can hit httptest.Server, which listens on 127.0.0.1 — an address
// safeDialContext blocks by design (see TestIsBlockedIP). Restores the real
// SSRF-checking dialer afterward regardless of test outcome.
func withPlainDialer(t *testing.T) {
	t.Helper()
	orig := fetchDialContext
	fetchDialContext = (&net.Dialer{}).DialContext
	t.Cleanup(func() { fetchDialContext = orig })
}

func newFetchRegistry(t *testing.T, cfg FetchURLConfig) *Registry {
	t.Helper()
	dir := t.TempDir()
	root, err := security.NewRoot(dir)
	must(t, err)
	return NewRegistry(root, DefaultLimits(), false, cfg, &fakeEmbedder{}, "fake-embed-model")
}

func TestIsBlockedIP(t *testing.T) {
	blocked := []string{
		"127.0.0.1", "::1", // loopback
		"10.0.0.1", "172.16.0.1", "192.168.1.1", // private
		"169.254.169.254", "fe80::1", // link-local (incl. cloud metadata)
		"0.0.0.0", // unspecified
	}
	for _, s := range blocked {
		ip := net.ParseIP(s)
		if ip == nil {
			t.Fatalf("ParseIP(%q) returned nil", s)
		}
		if !isBlockedIP(ip) {
			t.Errorf("isBlockedIP(%q) = false, want true", s)
		}
	}

	allowed := []string{"8.8.8.8", "1.1.1.1", "93.184.216.34"}
	for _, s := range allowed {
		ip := net.ParseIP(s)
		if ip == nil {
			t.Fatalf("ParseIP(%q) returned nil", s)
		}
		if isBlockedIP(ip) {
			t.Errorf("isBlockedIP(%q) = true, want false", s)
		}
	}
}

func TestFetchURLDisabledRefuses(t *testing.T) {
	r := newFetchRegistry(t, FetchURLConfig{Enabled: false})
	_, err := call(t, r, "fetch_url", map[string]any{"url": "https://example.com"})
	if err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Fatalf("expected disabled error, got: %v", err)
	}
}

func TestFetchURLNotAdvertisedWhenDisabled(t *testing.T) {
	r := newFetchRegistry(t, FetchURLConfig{Enabled: false})
	for _, d := range r.Definitions() {
		if d.Function.Name == "fetch_url" {
			t.Fatal("fetch_url should not be advertised when disabled")
		}
	}
}

func TestFetchURLBlocksLoopbackEvenWhenEnabled(t *testing.T) {
	// Deliberately do NOT call withPlainDialer here: this confirms the SSRF
	// guard itself (not just the enabled flag) rejects loopback targets.
	r := newFetchRegistry(t, FetchURLConfig{Enabled: true, HTTPSOnly: false, MaxBytes: 1 << 20, TimeoutSeconds: 5})
	out, err := call(t, r, "fetch_url", map[string]any{"url": "http://127.0.0.1:1/"})
	if err != nil {
		t.Fatalf("unexpected Go error: %v", err)
	}
	if !strings.Contains(out.Text, "HTTP 0") || !strings.Contains(out.Text, "blocked") {
		t.Fatalf("expected blocked-address result, got: %q", out.Text)
	}
}

func TestFetchURLRejectsPlainHTTPWhenHTTPSOnly(t *testing.T) {
	r := newFetchRegistry(t, FetchURLConfig{Enabled: true, HTTPSOnly: true, MaxBytes: 1 << 20, TimeoutSeconds: 5})
	out, err := call(t, r, "fetch_url", map[string]any{"url": "http://example.com"})
	if err != nil {
		t.Fatalf("unexpected Go error: %v", err)
	}
	if !strings.Contains(out.Text, "HTTP 0") || !strings.Contains(out.Text, "fetch_url_https_only") {
		t.Fatalf("expected https-only rejection, got: %q", out.Text)
	}
}

func TestCheckRedirectHTTPSOnlyBlocksDowngrade(t *testing.T) {
	httpsReq := &http.Request{URL: &url.URL{Scheme: "https", Host: "example.com"}}
	if err := checkRedirectHTTPSOnly(httpsReq, nil); err != nil {
		t.Errorf("https redirect target should be allowed, got error: %v", err)
	}

	httpReq := &http.Request{URL: &url.URL{Scheme: "http", Host: "example.com"}}
	if err := checkRedirectHTTPSOnly(httpReq, nil); err == nil {
		t.Error("http redirect target should be blocked, got nil error")
	}
}

func TestCheckRedirectHTTPSOnlyCapsRedirectChain(t *testing.T) {
	req := &http.Request{URL: &url.URL{Scheme: "https", Host: "example.com"}}
	via := make([]*http.Request, 10)
	if err := checkRedirectHTTPSOnly(req, via); err == nil {
		t.Error("expected redirect chain to be capped at 10, got nil error")
	}
}

// TestFetchURLIntegrationHTTPSOnlyOverPlainServer confirms that, end to end,
// a plain (non-TLS) target URL is rejected before any request is sent when
// fetch_url_https_only is enabled — the scheme check that guards
// checkRedirectHTTPSOnly's redirect-time logic.
func TestFetchURLIntegrationHTTPSOnlyOverPlainServer(t *testing.T) {
	withPlainDialer(t)

	reached := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		reached = true
		w.Write([]byte("should never be reached"))
	}))
	defer srv.Close()

	r := newFetchRegistry(t, FetchURLConfig{Enabled: true, HTTPSOnly: true, MaxBytes: 1 << 20, TimeoutSeconds: 5})
	out, err := call(t, r, "fetch_url", map[string]any{"url": srv.URL}) // srv.URL is http://
	if err != nil {
		t.Fatalf("unexpected Go error: %v", err)
	}
	if reached {
		t.Fatal("plain-http server should never have been contacted")
	}
	if !strings.Contains(out.Text, "fetch_url_https_only") {
		t.Fatalf("expected https-only rejection, got: %q", out.Text)
	}
}

func TestFetchURLTruncatesLargeExtractedText(t *testing.T) {
	withPlainDialer(t)

	big := strings.Repeat("word ", 20000) // well over fetchMaxExtractedBytes
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte(big))
	}))
	defer srv.Close()

	r := newFetchRegistry(t, FetchURLConfig{Enabled: true, HTTPSOnly: false, MaxBytes: 10 << 20, TimeoutSeconds: 5})
	out, err := call(t, r, "fetch_url", map[string]any{"url": srv.URL})
	if err != nil {
		t.Fatalf("unexpected Go error: %v", err)
	}
	if !strings.Contains(out.Text, "[content truncated]") {
		t.Fatalf("expected truncation note in output")
	}
	if len(out.Text) > fetchMaxExtractedBytes+200 {
		t.Fatalf("output too large: %d bytes", len(out.Text))
	}
}

func TestFetchURLSniffsMislabeledHTML(t *testing.T) {
	withPlainDialer(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// No Content-Type header set at all, mimicking a misconfigured server.
		w.Write([]byte("<!DOCTYPE html><html><body><h1>Title</h1><p>Body text.</p></body></html>"))
	}))
	defer srv.Close()

	r := newFetchRegistry(t, FetchURLConfig{Enabled: true, HTTPSOnly: false, MaxBytes: 1 << 20, TimeoutSeconds: 5})
	out, err := call(t, r, "fetch_url", map[string]any{"url": srv.URL})
	if err != nil {
		t.Fatalf("unexpected Go error: %v", err)
	}
	if strings.Contains(out.Text, "<html>") || strings.Contains(out.Text, "<body>") {
		t.Fatalf("expected HTML tags to be stripped/converted, got: %q", out.Text)
	}
	if !strings.Contains(out.Text, "Title") {
		t.Fatalf("expected extracted text to contain page content, got: %q", out.Text)
	}
}

func TestFetchURLSurfacesNon2xxWithoutGoError(t *testing.T) {
	withPlainDialer(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	r := newFetchRegistry(t, FetchURLConfig{Enabled: true, HTTPSOnly: false, MaxBytes: 1 << 20, TimeoutSeconds: 5})
	out, err := call(t, r, "fetch_url", map[string]any{"url": srv.URL})
	if err != nil {
		t.Fatalf("unexpected Go error: %v", err)
	}
	if !strings.Contains(out.Text, "HTTP 404") {
		t.Fatalf("expected HTTP 404 in result, got: %q", out.Text)
	}
}

// Ensure context cancellation still surfaces as a clean result, not a panic.
func TestFetchURLRespectsContextCancellation(t *testing.T) {
	withPlainDialer(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte("ok"))
	}))
	defer srv.Close()

	r := newFetchRegistry(t, FetchURLConfig{Enabled: true, HTTPSOnly: false, MaxBytes: 1 << 20, TimeoutSeconds: 5})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := r.Call(ctx, "fetch_url", map[string]any{"url": srv.URL})
	if err != nil {
		t.Fatalf("unexpected Go error (should be surfaced in result text): %v", err)
	}
}
