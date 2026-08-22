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

func TestLoadModelRequiresRegisteredBackend(t *testing.T) {
	if _, err := LoadModel(context.Background(), ModelOptions{Backend: "missing"}); err == nil {
		t.Fatal("expected missing backend error")
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
