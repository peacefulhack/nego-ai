package llama

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/modelinfo"
)

const BackendName = "llama.cpp"

func init() {
	_ = nego.RegisterBackend(BackendName, Backend{})
}

type Backend struct {
	Command string
}

func (b Backend) Info() nego.BackendInfo {
	return nego.BackendInfo{
		Name:         BackendName,
		Description:  "Local llama.cpp runner for GGUF models using llama-cli.",
		Capabilities: []string{"generate", "chat", "stream_chat"},
		Required:     []string{"path", "llama-cli or NEGO_LLAMA_CLI"},
		Options: []nego.BackendOption{
			{Name: "threads", Description: "CPU thread count passed as -t"},
			{Name: "ctx_size", Description: "context size passed as -c"},
			{Name: "gpu_layers", Description: "GPU layer count passed as -ngl"},
			{Name: "template_path", Description: "directory or file path for chat template sidecars"},
		},
	}
}

func (b Backend) Load(_ context.Context, opts nego.ModelOptions) (nego.Model, error) {
	if opts.Path == "" {
		return nil, fmt.Errorf("model path is required for %s backend", BackendName)
	}
	modelPath, err := modelinfo.ResolveRuntimeFile(opts.Path, "gguf")
	if err != nil {
		return nil, fmt.Errorf("resolve llama.cpp model: %w", err)
	}
	command := b.Command
	if command == "" {
		command = os.Getenv("NEGO_LLAMA_CLI")
	}
	if command == "" {
		command = "llama-cli"
	}
	if err := validateCommand(command); err != nil {
		return nil, err
	}
	return &Model{command: command, modelPath: modelPath, promptPath: promptPath(opts.Path, modelPath, opts.Options), options: opts.Options}, nil
}

type Model struct {
	command    string
	modelPath  string
	promptPath string
	options    map[string]string
}

func (m *Model) Generate(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	cmd := exec.CommandContext(ctx, m.command, m.args(req.Prompt, req)...)
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
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		return nil, err
	}
	out, err := m.Generate(ctx, nego.GenerateRequest{
		Prompt:      prompt,
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
	ctx, cancel := context.WithCancel(ctx)
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		cancel()
		return nil, err
	}
	genReq := nego.GenerateRequest{
		Prompt:      prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
		Seed:        req.Seed,
	}
	cmd := exec.CommandContext(ctx, m.command, m.args(genReq.Prompt, genReq)...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		cancel()
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, fmt.Errorf("llama.cpp command failed: %s", msg)
	}
	stream := newProcessStream(ctx, stdout, cmd, cancel, &stderr)
	go stream.read()
	return stream, nil
}

func (m *Model) Close() error {
	return nil
}

func (m *Model) renderChatPrompt(messages []nego.Message) (string, error) {
	if m.promptPath == "" {
		return chatPrompt(messages), nil
	}
	return chattemplate.Render(m.promptPath, messages, chattemplate.Options{AddGenerationPrompt: true})
}

func chatPrompt(messages []nego.Message) string {
	var b strings.Builder
	for _, message := range messages {
		fmt.Fprintf(&b, "%s: %s\n", strings.ToUpper(string(message.Role)), message.Content)
	}
	b.WriteString("ASSISTANT: ")
	return b.String()
}

func promptPath(inputPath, modelPath string, options map[string]string) string {
	if value := options["template_path"]; value != "" {
		return value
	}
	if inputPath != "" {
		if info, err := os.Stat(inputPath); err == nil && info.IsDir() {
			return inputPath
		}
	}
	if modelPath != "" {
		return filepath.Dir(modelPath)
	}
	return ""
}

func (m *Model) args(prompt string, req nego.GenerateRequest) []string {
	args := []string{"-m", m.modelPath, "-p", prompt}
	if req.MaxTokens > 0 {
		args = append(args, "-n", strconv.Itoa(req.MaxTokens))
	}
	if req.Temperature > 0 {
		args = append(args, "--temp", strconv.FormatFloat(req.Temperature, 'f', -1, 64))
	}
	if req.TopP > 0 {
		args = append(args, "--top-p", strconv.FormatFloat(req.TopP, 'f', -1, 64))
	}
	if req.Seed != 0 {
		args = append(args, "--seed", strconv.FormatInt(req.Seed, 10))
	}
	for _, stop := range req.Stop {
		if stop != "" {
			args = append(args, "--reverse-prompt", stop)
		}
	}
	if value := m.options["threads"]; value != "" {
		args = append(args, "-t", value)
	}
	if value := m.options["ctx_size"]; value != "" {
		args = append(args, "-c", value)
	}
	if value := m.options["gpu_layers"]; value != "" {
		args = append(args, "-ngl", value)
	}
	return args
}

type stream struct {
	ctx    context.Context
	stdout io.Reader
	tokens chan nego.Token
	done   chan struct{}
	cancel context.CancelFunc
	cmd    *exec.Cmd
	stderr *bytes.Buffer
	mu     sync.Mutex
	err    error
}

func newProcessStream(ctx context.Context, stdout io.Reader, cmd *exec.Cmd, cancel context.CancelFunc, stderr *bytes.Buffer) *stream {
	return &stream{
		ctx:    ctx,
		stdout: stdout,
		tokens: make(chan nego.Token, 8),
		done:   make(chan struct{}),
		cancel: cancel,
		cmd:    cmd,
		stderr: stderr,
	}
}

func (s *stream) Tokens() <-chan nego.Token {
	return s.tokens
}

func (s *stream) Err() error {
	<-s.done
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *stream) Close() error {
	s.cancel()
	<-s.done
	return nil
}

func (s *stream) read() {
	defer close(s.tokens)
	defer close(s.done)
	defer s.cancel()

	buf := make([]byte, 4096)
	for {
		n, readErr := s.stdout.Read(buf)
		if n > 0 {
			token := nego.Token{Text: string(buf[:n])}
			select {
			case s.tokens <- token:
			case <-s.ctx.Done():
				_ = s.cmd.Wait()
				return
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
				s.setErr(readErr)
			}
			break
		}
	}
	if err := s.cmd.Wait(); err != nil {
		msg := strings.TrimSpace(s.stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		s.setErr(fmt.Errorf("llama.cpp command failed: %s", msg))
	}
}

func (s *stream) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}

func validateCommand(command string) error {
	if command == "" {
		return fmt.Errorf("llama.cpp command is required")
	}
	if filepath.IsAbs(command) || strings.ContainsAny(command, `/\`) {
		if _, err := os.Stat(command); err != nil {
			return fmt.Errorf("llama.cpp command %q is not available; set NEGO_LLAMA_CLI or install llama-cli", command)
		}
		return nil
	}
	if _, err := exec.LookPath(command); err != nil {
		return fmt.Errorf("llama.cpp command %q is not available; set NEGO_LLAMA_CLI or install llama-cli", command)
	}
	return nil
}
