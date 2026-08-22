package llama

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
)

func TestGenerateUsesLlamaCommand(t *testing.T) {
	command := fakeLlamaCommand(t)
	model, err := Backend{Command: command}.Load(context.Background(), nego.ModelOptions{Path: "model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	out, err := model.Generate(context.Background(), nego.GenerateRequest{Prompt: "hello", MaxTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.Text, "fake llama output") || !strings.Contains(out.Text, "hello") {
		t.Fatalf("unexpected output: %q", out.Text)
	}
}

func TestChatUsesGeneratedOutput(t *testing.T) {
	command := fakeLlamaCommand(t)
	model, err := Backend{Command: command}.Load(context.Background(), nego.ModelOptions{Path: "model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := model.Chat(context.Background(), nego.ChatRequest{Messages: []nego.Message{{Role: nego.RoleUser, Content: "hello"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Role != nego.RoleAssistant || !strings.Contains(resp.Message.Content, "fake llama output") {
		t.Fatalf("unexpected response: %#v", resp)
	}
}

func TestGeneratePassesRuntimeOptions(t *testing.T) {
	command := fakeLlamaCommand(t)
	model, err := Backend{Command: command}.Load(context.Background(), nego.ModelOptions{
		Path: "model.gguf",
		Options: map[string]string{
			"threads":    "4",
			"ctx_size":   "2048",
			"gpu_layers": "20",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := model.Generate(context.Background(), nego.GenerateRequest{
		Prompt:      "hello",
		MaxTokens:   8,
		Temperature: 0.7,
		TopP:        0.9,
		Stop:        []string{"END"},
		Seed:        42,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"-n 8", "--temp 0.7", "--top-p 0.9", "--seed 42", "--reverse-prompt END", "-t 4", "-c 2048", "-ngl 20"} {
		if !strings.Contains(out.Text, want) {
			t.Fatalf("expected %q in command output: %q", want, out.Text)
		}
	}
}

func TestStreamChatStreamsProcessOutput(t *testing.T) {
	command := fakeLlamaCommand(t)
	model, err := Backend{Command: command}.Load(context.Background(), nego.ModelOptions{Path: "model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := model.StreamChat(context.Background(), nego.ChatRequest{
		Messages:  []nego.Message{{Role: nego.RoleUser, Content: "hello"}},
		MaxTokens: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	var got strings.Builder
	for token := range stream.Tokens() {
		got.WriteString(token.Text)
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got.String(), "fake llama output") || !strings.Contains(got.String(), "USER: hello") {
		t.Fatalf("unexpected stream output: %q", got.String())
	}
}

func TestLoadReportsMissingCommand(t *testing.T) {
	_, err := Backend{Command: "definitely-missing-nego-llama-command"}.Load(context.Background(), nego.ModelOptions{Path: "model.gguf"})
	if err == nil {
		t.Fatal("expected missing command error")
	}
	if !strings.Contains(err.Error(), "set NEGO_LLAMA_CLI") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func fakeLlamaCommand(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, "fake-llama.bat")
		if err := os.WriteFile(path, []byte("@echo off\necho fake llama output %*\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	path := filepath.Join(dir, "fake-llama.sh")
	if err := os.WriteFile(path, []byte("#!/bin/sh\necho fake llama output \"$@\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}
