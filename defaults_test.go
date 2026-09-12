package nego

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/gakon/nego-ai/training"
)

func TestApplyGenerationDefaults(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(`{"max_new_tokens":32,"temperature":0.7,"top_p":0.8,"repetition_penalty":1.1,"stop_strings":["</s>"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	req := GenerateRequest{}
	if err := ApplyGenerationDefaults(dir, &req); err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 32 || req.Temperature != 0.7 || req.TopP != 0.8 || req.RepeatPenalty != 1.1 || len(req.Stop) != 1 || req.Stop[0] != "</s>" {
		t.Fatalf("unexpected request: %#v", req)
	}
}

func TestApplyGenerationDefaultsUsesParentDirForModelFile(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(`{"max_new_tokens":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	req := GenerateRequest{}
	if err := ApplyGenerationDefaults(modelPath, &req); err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 8 {
		t.Fatalf("MaxTokens = %d", req.MaxTokens)
	}
}

func TestApplyGenerationDefaultsDoesNotOverrideExplicitValues(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(`{"max_new_tokens":32,"temperature":0.7,"stop_strings":["</s>"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	req := GenerateRequest{MaxTokens: 4, Temperature: 0.2, Stop: []string{"END"}}
	if err := ApplyGenerationDefaults(dir, &req); err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 4 || req.Temperature != 0.2 || len(req.Stop) != 1 || req.Stop[0] != "END" {
		t.Fatalf("unexpected request: %#v", req)
	}
}

func TestLoadModelExpandsNativeTrainingManifest(t *testing.T) {
	backend := &manifestCaptureBackend{}
	backendName := testBackendName("manifest")
	if err := RegisterBackend(backendName, backend); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	outputDir := filepath.Join(dir, "adapter")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := training.NativeManifest{
		Version:            1,
		Type:               "nego-native-adapter",
		BaseModel:          filepath.Join(dir, "models", "qwen3"),
		AdapterPath:        filepath.Join(outputDir, "adapter.json"),
		RecommendedBackend: backendName,
		RuntimeOptions:     map[string]string{"adapter_path": filepath.Join(outputDir, "adapter.json"), "foo": "bar"},
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "manifest.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	model, err := LoadModel(context.Background(), ModelOptions{Path: outputDir, Options: map[string]string{"foo": "override"}})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if backend.opts.Path != manifest.BaseModel || backend.opts.Backend != backendName {
		t.Fatalf("unexpected options: %#v", backend.opts)
	}
	if backend.opts.Options["adapter_path"] != manifest.AdapterPath || backend.opts.Options["foo"] != "override" {
		t.Fatalf("unexpected runtime options: %#v", backend.opts.Options)
	}
}

type manifestCaptureBackend struct {
	opts ModelOptions
}

func (b *manifestCaptureBackend) Load(_ context.Context, opts ModelOptions) (Model, error) {
	b.opts = opts
	return manifestTestModel{}, nil
}

type manifestTestModel struct{}

func (manifestTestModel) Generate(context.Context, GenerateRequest) (*GenerateOutput, error) {
	return &GenerateOutput{}, nil
}

func (manifestTestModel) Chat(context.Context, ChatRequest) (*ChatResponse, error) {
	return &ChatResponse{}, nil
}

func (manifestTestModel) StreamChat(context.Context, ChatRequest) (Stream, error) {
	return nil, nil
}

func (manifestTestModel) Close() error {
	return nil
}
