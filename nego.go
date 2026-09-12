package nego

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/modelinfo"
	"github.com/gakon/nego-ai/training"
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

type StreamingGenerator interface {
	StreamGenerate(ctx context.Context, req GenerateRequest) (Stream, error)
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

func StreamGenerate(ctx context.Context, model Model, req GenerateRequest) (Stream, error) {
	if model == nil {
		return nil, fmt.Errorf("model is required")
	}
	streamer, ok := model.(StreamingGenerator)
	if ok {
		return streamer.StreamGenerate(ctx, req)
	}
	out, err := model.Generate(ctx, req)
	if err != nil {
		return nil, err
	}
	tokens := make(chan Token, 1)
	if out != nil && out.Text != "" {
		tokens <- Token{Text: out.Text}
	}
	close(tokens)
	return staticStream{tokens: tokens}, nil
}

func ApplyGenerationDefaults(path string, req *GenerateRequest) error {
	if req == nil {
		return fmt.Errorf("generate request is nil")
	}
	cfg, err := modelinfo.LoadGenerationConfig(path)
	if err != nil || cfg == nil {
		return err
	}
	applyGenerationConfigToGenerateRequest(cfg, req)
	return nil
}

func ApplyChatGenerationDefaults(path string, req *ChatRequest) error {
	if req == nil {
		return fmt.Errorf("chat request is nil")
	}
	cfg, err := modelinfo.LoadGenerationConfig(path)
	if err != nil || cfg == nil {
		return err
	}
	if req.MaxTokens == 0 && cfg.MaxNewTokens != nil {
		req.MaxTokens = *cfg.MaxNewTokens
	}
	if req.Temperature == 0 && cfg.Temperature != nil {
		req.Temperature = *cfg.Temperature
	}
	if req.TopP == 0 && cfg.TopP != nil {
		req.TopP = *cfg.TopP
	}
	if req.RepeatPenalty == 0 && cfg.RepetitionPenalty != nil {
		req.RepeatPenalty = *cfg.RepetitionPenalty
	}
	if len(req.Stop) == 0 && len(cfg.StopStrings) > 0 {
		req.Stop = append([]string(nil), cfg.StopStrings...)
	}
	return nil
}

func applyGenerationConfigToGenerateRequest(cfg *modelinfo.GenerationConfig, req *GenerateRequest) {
	if req.MaxTokens == 0 && cfg.MaxNewTokens != nil {
		req.MaxTokens = *cfg.MaxNewTokens
	}
	if req.Temperature == 0 && cfg.Temperature != nil {
		req.Temperature = *cfg.Temperature
	}
	if req.TopP == 0 && cfg.TopP != nil {
		req.TopP = *cfg.TopP
	}
	if req.RepeatPenalty == 0 && cfg.RepetitionPenalty != nil {
		req.RepeatPenalty = *cfg.RepetitionPenalty
	}
	if len(req.Stop) == 0 && len(cfg.StopStrings) > 0 {
		req.Stop = append([]string(nil), cfg.StopStrings...)
	}
}

type staticStream struct {
	tokens <-chan Token
}

func (s staticStream) Tokens() <-chan Token {
	return s.tokens
}

func (staticStream) Err() error {
	return nil
}

func (staticStream) Close() error {
	return nil
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
	expanded, err := expandNativeManifestOptions(opts)
	if err != nil {
		return nil, err
	}
	opts = expanded
	if opts.Backend == "" {
		backend, err := ResolveBackend(opts)
		if err != nil {
			return nil, err
		}
		opts.Backend = backend
	}
	backendRegistry.RLock()
	backend := backendRegistry.backends[opts.Backend]
	backendRegistry.RUnlock()
	if backend == nil {
		return nil, fmt.Errorf("backend %q is not registered", opts.Backend)
	}
	return backend.Load(ctx, opts)
}

func ResolveBackend(opts ModelOptions) (string, error) {
	expanded, err := expandNativeManifestOptions(opts)
	if err != nil {
		return "", err
	}
	opts = expanded
	if opts.Backend != "" {
		return opts.Backend, nil
	}
	if strings.TrimSpace(opts.Endpoint) != "" || strings.TrimSpace(opts.Model) != "" {
		return "openai-compatible", nil
	}
	if strings.TrimSpace(opts.Path) == "" {
		return "", fmt.Errorf("model path or remote model endpoint is required")
	}
	artifact, err := modelinfo.Resolve(opts.Path)
	if err != nil {
		return "", err
	}
	if artifact.RecommendedRunBackend != "" {
		return artifact.RecommendedRunBackend, nil
	}
	return "", modelinfo.FormatResolveError(opts.Path, artifact)
}

func expandNativeManifestOptions(opts ModelOptions) (ModelOptions, error) {
	if strings.TrimSpace(opts.Path) == "" {
		return opts, nil
	}
	manifest, err := training.LoadNativeManifest(opts.Path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return opts, nil
		}
		return opts, fmt.Errorf("load native training manifest: %w", err)
	}
	if manifest.BaseModel != "" {
		opts.Path = manifest.BaseModel
	}
	if opts.Backend == "" {
		opts.Backend = manifest.RecommendedBackend
	}
	if len(manifest.RuntimeOptions) > 0 {
		if opts.Options == nil {
			opts.Options = make(map[string]string, len(manifest.RuntimeOptions))
		}
		for key, value := range manifest.RuntimeOptions {
			if _, exists := opts.Options[key]; !exists && value != "" {
				opts.Options[key] = value
			}
		}
	}
	return opts, nil
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
