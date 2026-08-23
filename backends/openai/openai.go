package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	nego "github.com/gakon/nego-ai"
)

const BackendName = "openai-compatible"

func init() {
	_ = nego.RegisterBackend(BackendName, Backend{HTTPClient: http.DefaultClient})
}

type Backend struct {
	HTTPClient *http.Client
}

func (b Backend) Info() nego.BackendInfo {
	return nego.BackendInfo{
		Name:         BackendName,
		Description:  "Remote OpenAI-compatible HTTP backend for completions, chat, streaming, and embeddings.",
		Capabilities: []string{"generate", "chat", "stream_chat", "embeddings"},
		Required:     []string{"endpoint", "model"},
		Options: []nego.BackendOption{
			{Name: "api_key", Description: "optional bearer token for authenticated endpoints"},
		},
	}
}

func (b Backend) Load(_ context.Context, opts nego.ModelOptions) (nego.Model, error) {
	if opts.Endpoint == "" {
		return nil, fmt.Errorf("endpoint is required for %s backend", BackendName)
	}
	if opts.Model == "" {
		return nil, fmt.Errorf("model is required for %s backend", BackendName)
	}
	client := b.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Model{client: client, endpoint: strings.TrimRight(opts.Endpoint, "/"), model: opts.Model, apiKey: opts.APIKey}, nil
}

type Model struct {
	client   *http.Client
	endpoint string
	model    string
	apiKey   string
}

func (m *Model) Generate(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	body := map[string]any{
		"model":       m.model,
		"prompt":      req.Prompt,
		"max_tokens":  req.MaxTokens,
		"temperature": req.Temperature,
		"top_p":       req.TopP,
		"stop":        req.Stop,
	}
	if req.Seed != 0 {
		body["seed"] = req.Seed
	}
	var out struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := m.postJSON(ctx, "/v1/completions", body, &out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("completion response did not include choices")
	}
	return &nego.GenerateOutput{Text: out.Choices[0].Text}, nil
}

func (m *Model) Chat(ctx context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	body := map[string]any{
		"model":       m.model,
		"messages":    req.Messages,
		"max_tokens":  req.MaxTokens,
		"temperature": req.Temperature,
		"top_p":       req.TopP,
		"stop":        req.Stop,
	}
	if req.Seed != 0 {
		body["seed"] = req.Seed
	}
	var out struct {
		Choices []struct {
			Message nego.Message `json:"message"`
		} `json:"choices"`
	}
	if err := m.postJSON(ctx, "/v1/chat/completions", body, &out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("chat response did not include choices")
	}
	return &nego.ChatResponse{Message: out.Choices[0].Message}, nil
}

func (m *Model) StreamChat(ctx context.Context, req nego.ChatRequest) (nego.Stream, error) {
	body := map[string]any{
		"model":       m.model,
		"messages":    req.Messages,
		"max_tokens":  req.MaxTokens,
		"temperature": req.Temperature,
		"top_p":       req.TopP,
		"stop":        req.Stop,
		"stream":      true,
	}
	if req.Seed != 0 {
		body["seed"] = req.Seed
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.endpoint+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	m.addHeaders(httpReq)
	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, httpError(resp)
	}
	stream := newSSEStream(resp.Body)
	go stream.read()
	return stream, nil
}

func (m *Model) Embed(ctx context.Context, req nego.EmbeddingRequest) (*nego.EmbeddingResponse, error) {
	body := map[string]any{
		"model": m.model,
		"input": req.Input,
	}
	var out struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := m.postJSON(ctx, "/v1/embeddings", body, &out); err != nil {
		return nil, err
	}
	embeddings := make([][]float64, 0, len(out.Data))
	for _, item := range out.Data {
		embeddings = append(embeddings, item.Embedding)
	}
	return &nego.EmbeddingResponse{Embeddings: embeddings}, nil
}

func (m *Model) Close() error {
	return nil
}

func (m *Model) postJSON(ctx context.Context, path string, body any, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.endpoint+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	m.addHeaders(req)
	resp, err := m.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return httpError(resp)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (m *Model) addHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+m.apiKey)
	}
}

type sseStream struct {
	body   io.ReadCloser
	tokens chan nego.Token
	mu     sync.Mutex
	err    error
}

func newSSEStream(body io.ReadCloser) *sseStream {
	return &sseStream{body: body, tokens: make(chan nego.Token)}
}

func (s *sseStream) Tokens() <-chan nego.Token {
	return s.tokens
}

func (s *sseStream) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *sseStream) Close() error {
	return s.body.Close()
}

func (s *sseStream) read() {
	defer close(s.tokens)
	defer s.body.Close()
	scanner := bufio.NewScanner(s.body)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			return
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			s.setErr(err)
			return
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				s.tokens <- nego.Token{Text: choice.Delta.Content}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		s.setErr(err)
	}
}

func (s *sseStream) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = err
}

func httpError(resp *http.Response) error {
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	msg := strings.TrimSpace(string(data))
	if msg == "" {
		msg = resp.Status
	}
	return fmt.Errorf("openai-compatible backend returned %s: %s", resp.Status, msg)
}
