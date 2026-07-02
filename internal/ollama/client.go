// Package ollama implements a minimal HTTP client for a local Ollama server,
// covering the /api/chat and /api/tags endpoints needed for tool-calling
// conversations.
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks to a local Ollama server.
type Client struct {
	baseURL string
	http    *http.Client
}

// NewClient creates a client pointed at baseURL (e.g. http://localhost:11434).
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		http:    &http.Client{Timeout: 5 * time.Minute},
	}
}

// Message is one turn in a chat conversation.
type Message struct {
	Role      string     `json:"role"` // "system", "user", "assistant", "tool"
	Content   string     `json:"content"`
	Thinking  string     `json:"thinking,omitempty"` // model's reasoning trace, if the model/request supports it
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"` // set on role "tool" responses
	// Images holds raw base64-encoded image data (no data: URI prefix) for
	// multimodal/vision-capable models to inspect alongside Content.
	Images []string `json:"images,omitempty"`
}

// ToolCall is a single tool invocation requested by the model.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction describes the function name and arguments the model chose.
type ToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// Tool describes a callable tool using JSON-schema style parameters, matching
// Ollama's OpenAI-compatible tool definition format.
type Tool struct {
	Type     string       `json:"type"` // always "function"
	Function ToolFunction `json:"function"`
}

// ToolFunction is the function definition within a Tool.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ChatRequest is the body sent to POST /api/chat.
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream"`
	// Think requests the model's reasoning trace (returned via
	// Message.Thinking) on models that support it. Omitted entirely when nil
	// so models without thinking support are unaffected.
	Think *bool `json:"think,omitempty"`
}

// ChatResponse is the body returned by POST /api/chat, one per streamed line
// (or the single full response when stream:false).
type ChatResponse struct {
	Model      string  `json:"model"`
	Message    Message `json:"message"`
	Done       bool    `json:"done"`
	DoneReason string  `json:"done_reason,omitempty"`
	Error      string  `json:"error,omitempty"`
}

// Chat sends a non-streaming chat completion request.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	req.Stream = false

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encode chat request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build chat request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling ollama at %s: %w (is `ollama serve` running?)", c.baseURL, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read ollama response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(data))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(data, &chatResp); err != nil {
		return nil, fmt.Errorf("decode ollama response: %w", err)
	}
	if chatResp.Error != "" {
		return nil, fmt.Errorf("ollama error: %s", chatResp.Error)
	}
	return &chatResp, nil
}

// ChatStream sends a streaming chat completion request. onChunk is invoked
// once per line Ollama streams back, each carrying an incremental piece of
// content and/or thinking; it is intended for live output and may be nil.
// Tool calls, when the model requests them, typically arrive in their own
// chunk with done:false (not necessarily the final done:true chunk), so they
// are accumulated across the whole stream rather than read off the last one.
// ChatStream returns the fully accumulated message once the stream ends.
func (c *Client) ChatStream(ctx context.Context, req ChatRequest, onChunk func(ChatResponse)) (*ChatResponse, error) {
	req.Stream = true

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encode chat request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build chat request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling ollama at %s: %w (is `ollama serve` running?)", c.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(data))
	}

	var final ChatResponse
	var content, thinking strings.Builder
	var toolCalls []ToolCall
	doneSeen := false

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var chunk ChatResponse
		if err := json.Unmarshal(line, &chunk); err != nil {
			return nil, fmt.Errorf("decode ollama stream chunk: %w", err)
		}
		if chunk.Error != "" {
			return nil, fmt.Errorf("ollama error: %s", chunk.Error)
		}

		content.WriteString(chunk.Message.Content)
		thinking.WriteString(chunk.Message.Thinking)
		if len(chunk.Message.ToolCalls) > 0 {
			toolCalls = append(toolCalls, chunk.Message.ToolCalls...)
		}

		if onChunk != nil {
			onChunk(chunk)
		}

		if chunk.Done {
			final = chunk
			doneSeen = true
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read ollama stream: %w", err)
	}
	if !doneSeen {
		// The connection closed (server crash, network drop, proxy timeout,
		// etc.) before Ollama sent its final done:true chunk. Without this
		// check we'd silently return a zero-value/partial message as if it
		// were a successful, complete answer.
		return nil, fmt.Errorf("ollama stream ended without a final response (connection closed early?)")
	}

	final.Message.Content = content.String()
	final.Message.Thinking = thinking.String()
	final.Message.ToolCalls = toolCalls
	return &final, nil
}

// showResponse mirrors the fields we need from POST /api/show. ModelInfo is a
// flat map of family-prefixed keys (e.g. "llama.context_length",
// "gemma4.context_length") since the key name depends on the model
// architecture; there is no single fixed field name across model families.
type showResponse struct {
	Capabilities []string       `json:"capabilities"`
	ModelInfo    map[string]any `json:"model_info"`
}

// ModelInfo describes the properties of a model that braai cares about.
type ModelInfo struct {
	Capabilities []string
	// ContextLength is the model's context window in tokens, or 0 if it
	// could not be determined (e.g. an older Ollama server that doesn't
	// report model_info).
	ContextLength int
}

// HasCapability reports whether the model advertises the given capability
// string (e.g. "vision", "tools", "thinking").
func (m ModelInfo) HasCapability(name string) bool {
	for _, c := range m.Capabilities {
		if c == name {
			return true
		}
	}
	return false
}

// ShowModel returns capability and context-length information for a model
// via /api/show.
func (c *Client) ShowModel(ctx context.Context, model string) (ModelInfo, error) {
	body, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return ModelInfo{}, fmt.Errorf("encode show request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/show", bytes.NewReader(body))
	if err != nil {
		return ModelInfo{}, fmt.Errorf("build show request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("calling ollama at %s: %w (is `ollama serve` running?)", c.baseURL, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("read ollama response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return ModelInfo{}, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(data))
	}

	var show showResponse
	if err := json.Unmarshal(data, &show); err != nil {
		return ModelInfo{}, fmt.Errorf("decode show response: %w", err)
	}

	info := ModelInfo{Capabilities: show.Capabilities}
	for key, val := range show.ModelInfo {
		if !strings.HasSuffix(key, ".context_length") {
			continue
		}
		if f, ok := val.(float64); ok {
			info.ContextLength = int(f)
		}
		break
	}
	return info, nil
}

// tagsResponse mirrors GET /api/tags.
type tagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

// ListModels returns the names of models currently available on the server.
func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("build tags request: %w", err)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling ollama at %s: %w (is `ollama serve` running?)", c.baseURL, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read ollama response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(data))
	}

	var tags tagsResponse
	if err := json.Unmarshal(data, &tags); err != nil {
		return nil, fmt.Errorf("decode tags response: %w", err)
	}

	names := make([]string, 0, len(tags.Models))
	for _, m := range tags.Models {
		names = append(names, m.Name)
	}
	return names, nil
}

// embedRequest is the body sent to POST /api/embed.
type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// embedResponse mirrors the fields we need from POST /api/embed.
type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Error      string      `json:"error"`
}

// Embed returns one embedding vector per input string, in the same order,
// via POST /api/embed. Not every model or server supports this — a server
// started without embedding support returns a clear error message that is
// passed through as-is, since it already explains the fix (e.g. "This
// server does not support embeddings. Start it with `--embeddings`").
func (c *Client) Embed(ctx context.Context, model string, inputs []string) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{Model: model, Input: inputs})
	if err != nil {
		return nil, fmt.Errorf("encode embed request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build embed request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling ollama at %s: %w (is `ollama serve` running?)", c.baseURL, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read ollama response: %w", err)
	}

	var embed embedResponse
	if resp.StatusCode != http.StatusOK {
		// Error responses (e.g. 501 when the server wasn't started with
		// embedding support) still carry a JSON {"error": "..."} body with a
		// clear, actionable message; surface that instead of the raw body.
		if jsonErr := json.Unmarshal(data, &embed); jsonErr == nil && embed.Error != "" {
			return nil, fmt.Errorf("ollama error: %s", embed.Error)
		}
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(data))
	}

	if err := json.Unmarshal(data, &embed); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	if embed.Error != "" {
		return nil, fmt.Errorf("ollama error: %s", embed.Error)
	}
	if len(embed.Embeddings) != len(inputs) {
		return nil, fmt.Errorf("ollama returned %d embeddings for %d inputs", len(embed.Embeddings), len(inputs))
	}
	return embed.Embeddings, nil
}
