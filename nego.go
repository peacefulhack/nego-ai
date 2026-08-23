package nego

import (
	"context"
	"fmt"
	"sort"
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
	Options  map[string]string
}

type GenerateRequest struct {
	Prompt        string
	MaxTokens     int
	Temperature   float64
	TopP          float64
	RepeatPenalty float64
	Stop          []string
	Seed          int64
}

type GenerateOutput struct {
	Text string
}

type ChatRequest struct {
	Messages      []Message
	MaxTokens     int
	Temperature   float64
	TopP          float64
	RepeatPenalty float64
	Stop          []string
	Seed          int64
}

type ChatResponse struct {
	Message Message
}

type Token struct {
	Text string
}

type EmbeddingRequest struct {
	Input []string `json:"input"`
}

type EmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
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

type BackendOption struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

type BackendInfo struct {
	Name         string          `json:"name"`
	Description  string          `json:"description,omitempty"`
	Capabilities []string        `json:"capabilities,omitempty"`
	Required     []string        `json:"required,omitempty"`
	Options      []BackendOption `json:"options,omitempty"`
}

type BackendDescriber interface {
	Info() BackendInfo
}

type EmbeddingModel interface {
	Embed(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error)
}

func Embed(ctx context.Context, model Model, req EmbeddingRequest) (*EmbeddingResponse, error) {
	embedder, ok := model.(EmbeddingModel)
	if !ok {
		return nil, fmt.Errorf("model does not support embeddings")
	}
	return embedder.Embed(ctx, req)
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

func ListBackends() []BackendInfo {
	backendRegistry.RLock()
	defer backendRegistry.RUnlock()
	infos := make([]BackendInfo, 0, len(backendRegistry.backends))
	for name, backend := range backendRegistry.backends {
		infos = append(infos, backendInfo(name, backend))
	}
	sort.Slice(infos, func(i, j int) bool {
		return infos[i].Name < infos[j].Name
	})
	return infos
}

func BackendInfoByName(name string) (BackendInfo, bool) {
	backendRegistry.RLock()
	backend := backendRegistry.backends[name]
	backendRegistry.RUnlock()
	if backend == nil {
		return BackendInfo{}, false
	}
	return backendInfo(name, backend), true
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

func backendInfo(name string, backend Backend) BackendInfo {
	info := BackendInfo{Name: name}
	if describer, ok := backend.(BackendDescriber); ok {
		info = describer.Info()
		if info.Name == "" {
			info.Name = name
		}
	}
	return info
}
