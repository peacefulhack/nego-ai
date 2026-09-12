package native

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/modelinfo"
)

const BackendName = "native"

func init() {
	_ = nego.RegisterBackend(BackendName, Backend{})
}

type Backend struct{}

func (b Backend) Info() nego.BackendInfo {
	return nego.BackendInfo{
		Name:         BackendName,
		Description:  "Experimental pure-Go GGUF runtime foundation for local model loading, tokenization, and early CPU inference.",
		Capabilities: []string{"load_gguf", "inspect_tensors", "tokenize_gguf", "generate_experimental", "stream_generate", "chat_experimental", "stream_chat", "kv_cache", "float32_tensor_cache", "legacy_quant_dequant", "k_quant_dequant"},
		Required:     []string{"path"},
		Options: []nego.BackendOption{
			{Name: "template_path", Description: "directory or file path for chat template sidecars"},
			{Name: "adapter", Description: "token-bias adapter JSON produced by native training"},
			{Name: "adapter_path", Description: "alias for adapter"},
		},
	}
}

func (b Backend) Load(_ context.Context, opts nego.ModelOptions) (nego.Model, error) {
	modelPath, err := modelinfo.ResolveRuntimeFile(opts.Path, "gguf")
	if err != nil {
		return nil, fmt.Errorf("resolve native GGUF model: %w", err)
	}
	info, err := modelinfo.InspectGGUF(modelPath)
	if err != nil {
		return nil, fmt.Errorf("inspect native GGUF model: %w", err)
	}
	if len(info.Tensors) == 0 {
		return nil, fmt.Errorf("native GGUF model %q has no tensor directory", modelPath)
	}
	tensors, err := openTensorStore(modelPath, info)
	if err != nil {
		return nil, fmt.Errorf("open native tensor store: %w", err)
	}
	vocab, err := modelinfo.InspectGGUFVocab(modelPath)
	if err != nil {
		tensors.Close()
		return nil, fmt.Errorf("inspect native GGUF vocab: %w", err)
	}
	spec, tensorNames, err := buildModelSpec(info)
	if err != nil {
		tensors.Close()
		return nil, fmt.Errorf("build native model spec: %w", err)
	}
	manifest := buildTensorManifest(info, spec, tensorNames)
	adapter, err := loadAdapter(opts.Options)
	if err != nil {
		tensors.Close()
		return nil, err
	}
	return &Model{
		path:       modelPath,
		info:       info,
		vocab:      vocab,
		spec:       spec,
		names:      tensorNames,
		manifest:   manifest,
		tensors:    tensors,
		float32:    make(map[string]cachedFloat32Tensor),
		adapter:    adapter,
		promptPath: promptPath(opts.Path, modelPath, opts.Options),
	}, nil
}

type cachedFloat32Tensor struct {
	values []float32
	tensor modelinfo.GGUFTensor
}

type Model struct {
	path       string
	info       *modelinfo.GGUFInfo
	vocab      *modelinfo.GGUFVocab
	spec       ModelSpec
	names      TensorNames
	manifest   TensorManifestReport
	tensors    *tensorStore
	float32    map[string]cachedFloat32Tensor
	tensorMu   sync.Mutex
	adapter    *adapters.TokenBiasAdapter
	promptPath string
}

func (m *Model) Generate(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	return m.generateText(ctx, req)
}

func (m *Model) StreamGenerate(ctx context.Context, req nego.GenerateRequest) (nego.Stream, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	stream := newNativeStream(cancel)
	go stream.run(ctx, func(emit func(string) error) error {
		_, err := m.generateTextWithEmitter(ctx, req, emit)
		return err
	})
	return stream, nil
}

func (m *Model) Chat(ctx context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		return nil, err
	}
	out, err := m.generateText(ctx, nego.GenerateRequest{
		Prompt:        prompt,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
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
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return m.StreamGenerate(ctx, nego.GenerateRequest{
		Prompt:        prompt,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		RepeatPenalty: req.RepeatPenalty,
		Stop:          req.Stop,
		Seed:          req.Seed,
	})
}

func (m *Model) Close() error {
	m.tensorMu.Lock()
	defer m.tensorMu.Unlock()
	if m.tensors == nil {
		return nil
	}
	err := m.tensors.Close()
	m.tensors = nil
	m.float32 = nil
	return err
}

func (m *Model) Info() *modelinfo.GGUFInfo {
	return m.info
}

func (m *Model) Vocab() *modelinfo.GGUFVocab {
	return m.vocab
}

func (m *Model) Spec() ModelSpec {
	return m.spec
}

func (m *Model) TensorNames() TensorNames {
	return m.names
}

func (m *Model) TensorManifest() TensorManifestReport {
	return m.manifest
}

func (m *Model) Adapter() *adapters.TokenBiasAdapter {
	return m.adapter
}

func (m *Model) ReadTensor(name string) ([]byte, modelinfo.GGUFTensor, error) {
	if m.tensors == nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("native tensor store is closed")
	}
	return m.tensors.ReadTensor(name)
}

func (m *Model) TensorReader(name string) (*io.SectionReader, modelinfo.GGUFTensor, error) {
	if m.tensors == nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("native tensor store is closed")
	}
	return m.tensors.TensorReader(name)
}

func (m *Model) LoadTensorFloat32(name string) ([]float32, modelinfo.GGUFTensor, error) {
	values, tensor, err := m.loadTensorFloat32Shared(name)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	return cloneFloat32(values), tensor, nil
}

func (m *Model) loadTensorFloat32Shared(name string) ([]float32, modelinfo.GGUFTensor, error) {
	m.tensorMu.Lock()
	defer m.tensorMu.Unlock()
	if m.tensors == nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("native tensor store is closed")
	}
	if cached, ok := m.float32[name]; ok {
		return cached.values, cached.tensor, nil
	}
	data, tensor, err := m.tensors.ReadTensor(name)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	values, err := tensorFloat32(tensor, data)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	if m.float32 == nil {
		m.float32 = make(map[string]cachedFloat32Tensor)
	}
	m.float32[name] = cachedFloat32Tensor{values: values, tensor: tensor}
	return values, tensor, nil
}

func (m *Model) DecodeTokenIDs(ids []int) (string, error) {
	if m.vocab == nil {
		return "", fmt.Errorf("native GGUF vocab is not loaded")
	}
	return m.vocab.Decode(ids, modelinfo.DecodeOptions{SkipSpecial: true})
}

func (m *Model) EncodeText(text string) ([]int, error) {
	if m.vocab == nil {
		return nil, fmt.Errorf("native GGUF vocab is not loaded")
	}
	return m.vocab.Encode(text, modelinfo.EncodeOptions{})
}

func (m *Model) renderChatPrompt(messages []nego.Message) (string, error) {
	if m.promptPath == "" {
		return fallbackChatPrompt(messages), nil
	}
	return chattemplate.Render(m.promptPath, messages, chattemplate.Options{AddGenerationPrompt: true})
}

func (m *Model) inferenceError() error {
	return fmt.Errorf("native GGUF inference is not implemented yet for architecture %q with %d tensors; loaded %s without llama-cli, but transformer forward pass and sampling are still in progress", m.info.Architecture, len(m.info.Tensors), m.path)
}

func (m *Model) applyAdapter(logits []float32) []float32 {
	if m.adapter == nil {
		return logits
	}
	return m.adapter.Apply(logits)
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
		return nil, fmt.Errorf("load native adapter: %w", err)
	}
	return adapter, nil
}

func promptPath(inputPath, modelPath string, options map[string]string) string {
	if value := options["template_path"]; value != "" {
		return value
	}
	if inputPath != "" {
		if info, err := os.Stat(inputPath); err == nil && !info.IsDir() {
			return filepath.Dir(inputPath)
		}
		return inputPath
	}
	if modelPath != "" {
		return filepath.Dir(modelPath)
	}
	return ""
}

func fallbackChatPrompt(messages []nego.Message) string {
	var b strings.Builder
	for _, message := range messages {
		fmt.Fprintf(&b, "%s: %s\n", strings.ToUpper(string(message.Role)), message.Content)
	}
	b.WriteString("ASSISTANT: ")
	return b.String()
}
