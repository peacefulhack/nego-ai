package llama

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	nego "github.com/gakon/nego-ai"
)

const BackendName = "llama.cpp"

func init() {
	_ = nego.RegisterBackend(BackendName, Backend{})
}

type Backend struct {
	Command string
}

func (b Backend) Load(_ context.Context, opts nego.ModelOptions) (nego.Model, error) {
	if opts.Path == "" {
		return nil, fmt.Errorf("model path is required for %s backend", BackendName)
	}
	command := b.Command
	if command == "" {
		command = os.Getenv("NEGO_LLAMA_CLI")
	}
	if command == "" {
		command = "llama-cli"
	}
	return &Model{command: command, modelPath: opts.Path}, nil
}

type Model struct {
	command   string
	modelPath string
}

func (m *Model) Generate(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	args := []string{"-m", m.modelPath, "-p", req.Prompt}
	if req.MaxTokens > 0 {
		args = append(args, "-n", strconv.Itoa(req.MaxTokens))
	}
	cmd := exec.CommandContext(ctx, m.command, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, fmt.Errorf("llama.cpp command failed: %s", msg)
	}
	return &nego.GenerateOutput{Text: stdout.String()}, nil
}

func (m *Model) Chat(ctx context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	out, err := m.Generate(ctx, nego.GenerateRequest{
		Prompt:      chatPrompt(req.Messages),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
		Seed:        req.Seed,
	})
	if err != nil {
		return nil, err
	}
	return &nego.ChatResponse{Message: nego.Message{Role: nego.RoleAssistant, Content: out.Text}}, nil
}

func (m *Model) StreamChat(ctx context.Context, req nego.ChatRequest) (nego.Stream, error) {
	chat, err := m.Chat(ctx, req)
	if err != nil {
		return nil, err
	}
	ch := make(chan nego.Token, 1)
	ch <- nego.Token{Text: chat.Message.Content}
	close(ch)
	return stream{tokens: ch}, nil
}

func (m *Model) Close() error {
	return nil
}

func chatPrompt(messages []nego.Message) string {
	var b strings.Builder
	for _, message := range messages {
		fmt.Fprintf(&b, "%s: %s\n", strings.ToUpper(string(message.Role)), message.Content)
	}
	b.WriteString("ASSISTANT: ")
	return b.String()
}

type stream struct {
	tokens <-chan nego.Token
}

func (s stream) Tokens() <-chan nego.Token {
	return s.tokens
}

func (stream) Err() error {
	return nil
}

func (stream) Close() error {
	return nil
}
