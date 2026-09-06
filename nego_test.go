package nego

import (
	"context"
	"testing"
)

func TestLoadModelUsesRegisteredBackend(t *testing.T) {
	name := "mock-test-backend"
	if err := RegisterBackend(name, mockBackend{}); err != nil {
		t.Fatal(err)
	}
	model, err := LoadModel(context.Background(), ModelOptions{Backend: name, Path: "./model"})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	out, err := model.Generate(context.Background(), GenerateRequest{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if out.Text != "generated: hello" {
		t.Fatalf("Generate = %#v", out)
	}

	chat, err := model.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hello"}}})
	if err != nil {
		t.Fatal(err)
	}
	if chat.Message.Role != RoleAssistant || chat.Message.Content != "chat: hello" {
		t.Fatalf("Chat = %#v", chat)
	}

	stream, err := model.StreamChat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hello"}}})
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
	if text != "stream" {
		t.Fatalf("stream text = %q", text)
	}
}

func TestEmbedUsesOptionalCapability(t *testing.T) {
	resp, err := Embed(context.Background(), embeddingModel{}, EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) != 2 {
		t.Fatalf("unexpected embeddings: %#v", resp)
	}
	if _, err := Embed(context.Background(), mockModel{}, EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("expected unsupported embedding error")
	}
}

func TestLoadModelRequiresRegisteredBackend(t *testing.T) {
	if _, err := LoadModel(context.Background(), ModelOptions{Backend: "missing"}); err == nil {
		t.Fatal("expected missing backend error")
	}
}

func TestLoadModelAutoSelectsRemoteBackend(t *testing.T) {
	name := "openai-compatible"
	if _, ok := BackendInfoByName(name); !ok {
		if err := RegisterBackend(name, mockBackend{}); err != nil {
			t.Fatal(err)
		}
	}
	model, err := LoadModel(context.Background(), ModelOptions{
		Endpoint: "http://localhost:8080/v1",
		Model:    "local-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
}

func TestResolveBackendRequiresLocalOrRemoteTarget(t *testing.T) {
	if _, err := ResolveBackend(ModelOptions{}); err == nil {
		t.Fatal("expected target error")
	}
}

func TestBackendInfoDiscovery(t *testing.T) {
	name := "described-test-backend"
	if err := RegisterBackend(name, describedBackend{}); err != nil {
		t.Fatal(err)
	}
	info, ok := BackendInfoByName(name)
	if !ok {
		t.Fatal("expected backend info")
	}
	if info.Name != name || info.Description != "test backend" || len(info.Capabilities) != 1 {
		t.Fatalf("info = %#v", info)
	}
	found := false
	for _, backend := range ListBackends() {
		if backend.Name == name {
			found = true
		}
	}
	if !found {
		t.Fatalf("backend %q not found in list", name)
	}
}

type mockBackend struct{}

func (mockBackend) Load(context.Context, ModelOptions) (Model, error) {
	return mockModel{}, nil
}

type mockModel struct{}

func (mockModel) Generate(_ context.Context, req GenerateRequest) (*GenerateOutput, error) {
	return &GenerateOutput{Text: "generated: " + req.Prompt}, nil
}

func (mockModel) Chat(_ context.Context, req ChatRequest) (*ChatResponse, error) {
	content := ""
	if len(req.Messages) > 0 {
		content = req.Messages[len(req.Messages)-1].Content
	}
	return &ChatResponse{Message: Message{Role: RoleAssistant, Content: "chat: " + content}}, nil
}

func (mockModel) StreamChat(context.Context, ChatRequest) (Stream, error) {
	ch := make(chan Token, 1)
	ch <- Token{Text: "stream"}
	close(ch)
	return mockStream{tokens: ch}, nil
}

func (mockModel) Close() error {
	return nil
}

type mockStream struct {
	tokens <-chan Token
}

func (s mockStream) Tokens() <-chan Token {
	return s.tokens
}

func (mockStream) Err() error {
	return nil
}

func (mockStream) Close() error {
	return nil
}

type embeddingModel struct {
	mockModel
}

func (embeddingModel) Embed(context.Context, EmbeddingRequest) (*EmbeddingResponse, error) {
	return &EmbeddingResponse{Embeddings: [][]float64{{1, 0}}}, nil
}

type describedBackend struct {
	mockBackend
}

func (describedBackend) Info() BackendInfo {
	return BackendInfo{
		Description:  "test backend",
		Capabilities: []string{"generate"},
	}
}
