package nego

import (
	"os"
	"path/filepath"
	"testing"
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
