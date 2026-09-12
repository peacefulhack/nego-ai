package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

func TestModelsEndpoint(t *testing.T) {
	handler := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "test-model") {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestCompletionEndpoint(t *testing.T) {
	handler := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":"hello"}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var body struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Choices) != 1 || body.Choices[0].Text != "generated: hello" {
		t.Fatalf("unexpected body: %#v", body)
	}
}

func TestCompletionStreamEndpoint(t *testing.T) {
	handler := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"stream":true,"prompt":"hello"}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{"data:", `"object":"text_completion"`, `"text":"gen"`, "[DONE]"} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected %q in stream: %s", want, body)
		}
	}
}

func TestCompletionEndpointAppliesGenerationDefaults(t *testing.T) {
	model := &captureModel{}
	handler, err := NewHandler(HandlerOptions{
		ModelID: "test-model",
		Model:   model,
		Generation: &modelinfo.GenerationConfig{
			MaxNewTokens:      intServerPtr(12),
			Temperature:       floatServerPtr(0.4),
			TopK:              intServerPtr(20),
			TopP:              floatServerPtr(0.75),
			RepetitionPenalty: floatServerPtr(1.15),
			StopStrings:       []string{"END"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":"hello"}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	got := model.generateReq
	if got.MaxTokens != 12 || got.Temperature != 0.4 || got.TopK != 20 || got.TopP != 0.75 || got.RepeatPenalty != 1.15 || len(got.Stop) != 1 || got.Stop[0] != "END" {
		t.Fatalf("unexpected request: %#v", got)
	}
}

func TestCompletionEndpointKeepsExplicitZeroTemperature(t *testing.T) {
	model := &captureModel{}
	handler, err := NewHandler(HandlerOptions{
		ModelID:    "test-model",
		Model:      model,
		Generation: &modelinfo.GenerationConfig{Temperature: floatServerPtr(0.4)},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":"hello","temperature":0}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if model.generateReq.Temperature != 0 {
		t.Fatalf("Temperature = %v", model.generateReq.Temperature)
	}
}

func TestChatCompletionEndpoint(t *testing.T) {
	handler := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"content":"chat: hello"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestChatCompletionEndpointAppliesGenerationDefaults(t *testing.T) {
	model := &captureModel{}
	handler, err := NewHandler(HandlerOptions{
		ModelID: "test-model",
		Model:   model,
		Generation: &modelinfo.GenerationConfig{
			MaxNewTokens: intServerPtr(9),
			Temperature:  floatServerPtr(0.6),
			TopK:         intServerPtr(30),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	got := model.chatReq
	if got.MaxTokens != 9 || got.Temperature != 0.6 || got.TopK != 30 {
		t.Fatalf("unexpected request: %#v", got)
	}
}

func TestChatCompletionStreamEndpoint(t *testing.T) {
	handler := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"stream":true,"messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, "data:") || !strings.Contains(body, "[DONE]") {
		t.Fatalf("unexpected stream: %s", body)
	}
}

func TestEmbeddingsEndpoint(t *testing.T) {
	handler, err := NewHandler(HandlerOptions{ModelID: "test-model", Model: embeddingTestModel{}})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"input":["hello"]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"embedding":[1,0]`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestEmbeddingsEndpointRejectsUnsupportedModel(t *testing.T) {
	handler := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"input":["hello"]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func newTestHandler(t *testing.T) http.Handler {
	t.Helper()
	handler, err := NewHandler(HandlerOptions{ModelID: "test-model", Model: testModel{}})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

type testModel struct{}

func (testModel) Generate(_ context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	return &nego.GenerateOutput{Text: "generated: " + req.Prompt}, nil
}

func (testModel) StreamGenerate(context.Context, nego.GenerateRequest) (nego.Stream, error) {
	ch := make(chan nego.Token, 2)
	ch <- nego.Token{Text: "gen"}
	ch <- nego.Token{Text: "erated"}
	close(ch)
	return testStream{tokens: ch}, nil
}

func (testModel) Chat(_ context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	content := ""
	if len(req.Messages) > 0 {
		content = req.Messages[len(req.Messages)-1].Content
	}
	return &nego.ChatResponse{Message: nego.Message{Role: nego.RoleAssistant, Content: "chat: " + content}}, nil
}

func (testModel) StreamChat(context.Context, nego.ChatRequest) (nego.Stream, error) {
	ch := make(chan nego.Token, 2)
	ch <- nego.Token{Text: "hel"}
	ch <- nego.Token{Text: "lo"}
	close(ch)
	return testStream{tokens: ch}, nil
}

func (testModel) Close() error {
	return nil
}

type captureModel struct {
	testModel
	generateReq nego.GenerateRequest
	chatReq     nego.ChatRequest
}

func (m *captureModel) Generate(_ context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	m.generateReq = req
	return &nego.GenerateOutput{Text: "generated: " + req.Prompt}, nil
}

func (m *captureModel) Chat(_ context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	m.chatReq = req
	content := ""
	if len(req.Messages) > 0 {
		content = req.Messages[len(req.Messages)-1].Content
	}
	return &nego.ChatResponse{Message: nego.Message{Role: nego.RoleAssistant, Content: "chat: " + content}}, nil
}

func intServerPtr(value int) *int {
	return &value
}

func floatServerPtr(value float64) *float64 {
	return &value
}

type testStream struct {
	tokens <-chan nego.Token
}

func (s testStream) Tokens() <-chan nego.Token {
	return s.tokens
}

func (testStream) Err() error {
	return nil
}

func (testStream) Close() error {
	return nil
}

type embeddingTestModel struct {
	testModel
}

func (embeddingTestModel) Embed(context.Context, nego.EmbeddingRequest) (*nego.EmbeddingResponse, error) {
	return &nego.EmbeddingResponse{Embeddings: [][]float64{{1, 0}}}, nil
}
