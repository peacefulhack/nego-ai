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
		Description:  "Experimental pure-Go GGUF runtime foundation.",
		Capabilities: []string{"load_gguf", "inspect_tensors"},
		Required:     []string{"path"},
		Options: []nego.BackendOption{
			{Name: "template_path", Description: "directory or file path for chat template sidecars"},
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
	return &Model{
		path:       modelPath,
		info:       info,
		vocab:      vocab,
		spec:       spec,
		names:      tensorNames,
		manifest:   manifest,
		tensors:    tensors,
		float32:    make(map[string]cachedFloat32Tensor),
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
	promptPath string
}

func (m *Model) Generate(_ context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	return m.generateText(req)
}

func (m *Model) Chat(_ context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		return nil, err
	}
	out, err := m.generateText(nego.GenerateRequest{
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

func (m *Model) StreamChat(context.Context, nego.ChatRequest) (nego.Stream, error) {
	return nil, m.inferenceError()
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
