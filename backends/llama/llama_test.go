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
