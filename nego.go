package nego

import (
	"context"
	"fmt"
	"sync"

	"github.com/gakon/nego-ai/chattemplate"
)

type Role = chattemplate.Role
type Message = chattemplate.Message

const (
	RoleSystem    = chattemplate.RoleSystem
	RoleUser      = chattemplate.RoleUser
	RoleAssistant = chattemplate.RoleAssistant
)

type ModelOptions struct {
	Path     string
	Backend  string
	Endpoint string
	Model    string
	APIKey   string
}

type GenerateRequest struct {
	Prompt      string
	MaxTokens   int
	Temperature float64
	TopP        float64
	Stop        []string
	Seed        int64
}

type GenerateOutput struct {
	Text string
}

type ChatRequest struct {
	Messages    []Message
	MaxTokens   int
	Temperature float64
	TopP        float64
	Stop        []string
	Seed        int64
}

type ChatResponse struct {
	Message Message
}

type Token struct {
	Text string
}

type Stream interface {
	Tokens() <-chan Token
	Err() error
	Close() error
}

type Model interface {
	Generate(ctx context.Context, req GenerateRequest) (*GenerateOutput, error)
	Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)
	StreamChat(ctx context.Context, req ChatRequest) (Stream, error)
	Close() error
}

type Backend interface {
	Load(ctx context.Context, opts ModelOptions) (Model, error)
}

var backendRegistry = struct {
	sync.RWMutex
	backends map[string]Backend
}{backends: make(map[string]Backend)}

func RegisterBackend(name string, backend Backend) error {
	if name == "" {
		return fmt.Errorf("backend name is required")
	}
	if backend == nil {
		return fmt.Errorf("backend %q is nil", name)
	}
	backendRegistry.Lock()
	defer backendRegistry.Unlock()
	if _, exists := backendRegistry.backends[name]; exists {
		return fmt.Errorf("backend %q is already registered", name)
	}
	backendRegistry.backends[name] = backend
	return nil
}

func LoadModel(ctx context.Context, opts ModelOptions) (Model, error) {
	if opts.Backend == "" {
		return nil, fmt.Errorf("model backend is required")
	}
	backendRegistry.RLock()
	backend := backendRegistry.backends[opts.Backend]
	backendRegistry.RUnlock()
	if backend == nil {
		return nil, fmt.Errorf("backend %q is not registered", opts.Backend)
	}
	return backend.Load(ctx, opts)
}
