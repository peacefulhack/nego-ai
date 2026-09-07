package nativehf

import (
	"context"
	"fmt"
	"strings"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/modelinfo"
	"github.com/gakon/nego-ai/tokenizer"
)

const BackendName = "native-hf"

func init() {
	_ = nego.RegisterBackend(BackendName, Backend{})
}

type Backend struct{}

func (b Backend) Info() nego.BackendInfo {
	return nego.BackendInfo{
		Name:         BackendName,
		Description:  "Experimental pure-Go Hugging Face safetensors runtime loader.",
		Capabilities: []string{"load_safetensors", "inspect_tensors", "load_tokenizer", "load_adapter"},
		Required:     []string{"path"},
		Options: []nego.BackendOption{
			{Name: "adapter", Description: "token-bias adapter JSON produced by native training"},
			{Name: "adapter_path", Description: "alias for adapter"},
			{Name: "experimental_generation", Description: "enable guarded native-hf greedy generation for small compatible fixtures"},
		},
	}
}

func (b Backend) Load(_ context.Context, opts nego.ModelOptions) (nego.Model, error) {
	if strings.TrimSpace(opts.Path) == "" {
		return nil, fmt.Errorf("native-hf model path is required")
	}
	info, err := modelinfo.Inspect(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("inspect native-hf model: %w", err)
	}
	if info.Safetensors == nil {
		return nil, fmt.Errorf("native-hf requires safetensors weights")
	}
	if info.HFWeights == nil {
		return nil, fmt.Errorf("native-hf requires a Hugging Face weight manifest")
	}
	store, err := modelinfo.OpenSafetensors(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors store: %w", err)
	}
	tok, err := tokenizer.Load(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	adapter, err := loadAdapter(opts.Options)
	if err != nil {
		return nil, err
	}
	return &Model{
		path:      opts.Path,
		info:      info,
		spec:      info.HFSpec,
		store:     store,
		tokenizer: tok,
		adapter:   adapter,
		generate:  optionBool(opts.Options, "experimental_generation"),
	}, nil
}

type Model struct {
	path      string
	info      *modelinfo.Info
	spec      *modelinfo.HFModelSpec
	store     *modelinfo.SafetensorsStore
	tokenizer *tokenizer.Tokenizer
	adapter   *adapters.TokenBiasAdapter
	generate  bool
}

func (m *Model) Generate(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	if !m.generate {
		return nil, m.inferenceError()
	}
	return m.generateText(ctx, req)
}

func (m *Model) Chat(ctx context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		return nil, err
	}
	out, err := m.Generate(ctx, nego.GenerateRequest{
		Prompt:        prompt,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		RepeatPenalty: req.RepeatPenalty,
		Stop:          req.Stop,
		Seed:          req.Seed,
	})
	if err != nil {
		return nil, err
	}
	return &nego.ChatResponse{Message: nego.Message{Role: nego.RoleAssistant, Content: out.Text}}, nil
}

func (m *Model) StreamChat(ctx context.Context, req nego.ChatRequest) (nego.Stream, error) {
	resp, err := m.Chat(ctx, req)
	if err != nil {
		return nil, err
	}
	return newStaticStream(resp.Message.Content), nil
}

func (m *Model) Close() error {
	return nil
}

func (m *Model) Info() *modelinfo.Info {
	return m.info
}

func (m *Model) Spec() *modelinfo.HFModelSpec {
	return m.spec
}

func (m *Model) Store() *modelinfo.SafetensorsStore {
	return m.store
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tokenizer
}

func (m *Model) Adapter() *adapters.TokenBiasAdapter {
	return m.adapter
}

func (m *Model) TensorReader(name string) (*modelinfo.SafetensorsTensorReader, modelinfo.SafetensorsTensor, error) {
	if m.store == nil {
		return nil, modelinfo.SafetensorsTensor{}, fmt.Errorf("native-hf safetensors store is not loaded")
	}
	return m.store.TensorReader(name)
}

func (m *Model) ReadTensor(name string) ([]byte, modelinfo.SafetensorsTensor, error) {
	if m.store == nil {
		return nil, modelinfo.SafetensorsTensor{}, fmt.Errorf("native-hf safetensors store is not loaded")
	}
	return m.store.ReadTensor(name)
}

func (m *Model) LoadTensorFloat32(name string) ([]float32, modelinfo.SafetensorsTensor, error) {
	if m.store == nil {
		return nil, modelinfo.SafetensorsTensor{}, fmt.Errorf("native-hf safetensors store is not loaded")
	}
	return m.store.LoadTensorFloat32(name)
}

func (m *Model) inferenceError() error {
	ready := false
	missing := 0
	if m.info != nil && m.info.HFWeights != nil {
		ready = m.info.HFWeights.Ready
		missing = len(m.info.HFWeights.Missing)
	}
	return fmt.Errorf("native-hf production generation is not enabled for %q; loaded safetensors, tokenizer, and manifest (ready=%v missing_tensors=%d), and guarded greedy generation is available only for small compatible decoder fixtures with experimental_generation=true", m.path, ready, missing)
}

func loadAdapter(options map[string]string) (*adapters.TokenBiasAdapter, error) {
	if len(options) == 0 {
		return nil, nil
	}
	path := options["adapter"]
	if path == "" {
		path = options["adapter_path"]
	}
	if path == "" {
		return nil, nil
	}
	adapter, err := adapters.Load(path)
	if err != nil {
		return nil, fmt.Errorf("load native-hf adapter: %w", err)
	}
	return adapter, nil
}

func optionBool(options map[string]string, key string) bool {
	switch strings.ToLower(strings.TrimSpace(options[key])) {
	case "1", "t", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}

func (m *Model) renderChatPrompt(messages []nego.Message) (string, error) {
	if m.path != "" && hasChatTemplateFile(m.info) {
		prompt, err := chattemplate.Render(m.path, messages, chattemplate.Options{AddGenerationPrompt: true})
		if err != nil {
			return "", err
		}
		return prompt, nil
	}
	var b strings.Builder
	for _, message := range messages {
		fmt.Fprintf(&b, "%s: %s\n", strings.ToUpper(string(message.Role)), message.Content)
	}
	b.WriteString("ASSISTANT: ")
	return b.String(), nil
}

func hasChatTemplateFile(info *modelinfo.Info) bool {
	if info == nil {
		return false
	}
	for _, file := range info.Files {
		if file.Kind == "chat_template" || file.Kind == "tokenizer_config" {
			return true
		}
	}
	return false
}
