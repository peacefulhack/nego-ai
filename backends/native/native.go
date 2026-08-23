package native

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

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
	return &Model{
		path:       modelPath,
		info:       info,
		tensors:    tensors,
		promptPath: promptPath(opts.Path, modelPath, opts.Options),
	}, nil
}

type Model struct {
	path       string
	info       *modelinfo.GGUFInfo
	tensors    *tensorStore
	promptPath string
}

func (m *Model) Generate(context.Context, nego.GenerateRequest) (*nego.GenerateOutput, error) {
	return nil, m.inferenceError()
}

func (m *Model) Chat(_ context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	if _, err := m.renderChatPrompt(req.Messages); err != nil {
		return nil, err
	}
	return nil, m.inferenceError()
}

func (m *Model) StreamChat(context.Context, nego.ChatRequest) (nego.Stream, error) {
	return nil, m.inferenceError()
}

func (m *Model) Close() error {
	if m.tensors == nil {
		return nil
	}
	err := m.tensors.Close()
	m.tensors = nil
	return err
}

func (m *Model) Info() *modelinfo.GGUFInfo {
	return m.info
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
