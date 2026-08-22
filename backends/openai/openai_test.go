package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	nego "github.com/gakon/nego-ai"
)

func TestChatAndGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("missing auth header: %q", r.Header.Get("Authorization"))
		}
		switch r.URL.Path {
		case "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "hello"}}},
			})
		case "/v1/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"text": "generated"}},
			})
		case "/v1/embeddings":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"embedding": []float64{1, 0, 0}}},
			})
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	model, err := Backend{HTTPClient: server.Client()}.Load(context.Background(), nego.ModelOptions{
		Endpoint: server.URL,
		Model:    "test-model",
		APIKey:   "test-key",
	})
	if err != nil {
		t.Fatal(err)
	}
	chat, err := model.Chat(context.Background(), nego.ChatRequest{Messages: []nego.Message{{Role: nego.RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if chat.Message.Content != "hello" {
		t.Fatalf("chat = %#v", chat)
	}
	generated, err := model.Generate(context.Background(), nego.GenerateRequest{Prompt: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if generated.Text != "generated" {
		t.Fatalf("generated = %#v", generated)
	}
	embeddings, err := model.(nego.EmbeddingModel).Embed(context.Background(), nego.EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(embeddings.Embeddings) != 1 || len(embeddings.Embeddings[0]) != 3 {
		t.Fatalf("embeddings = %#v", embeddings)
	}
}

func TestStreamChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	model, err := Backend{HTTPClient: server.Client()}.Load(context.Background(), nego.ModelOptions{
		Endpoint: server.URL,
		Model:    "test-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := model.StreamChat(context.Background(), nego.ChatRequest{Messages: []nego.Message{{Role: nego.RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for token := range stream.Tokens() {
		text += token.Text
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("stream text = %q", text)
	}
}
