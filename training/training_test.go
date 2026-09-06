package training

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRunTrainingJob(t *testing.T) {
	result, err := Run(context.Background(), JobSpec{
		Name:    "test",
		Command: fakeTrainingCommand(t),
		Args:    []string{"hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Success || !strings.Contains(result.Stdout, "training hello") {
		t.Fatalf("result = %#v", result)
	}
}

func TestRunRequiresCommand(t *testing.T) {
	if _, err := Run(context.Background(), JobSpec{}); err == nil {
		t.Fatal("expected validation error")
	}
}

func TestValidateRejectsInvalidJobPaths(t *testing.T) {
	dir := t.TempDir()
	outputFile := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(outputFile, []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := Validate(JobSpec{
		Name:      "bad-output",
		Command:   "python",
		OutputDir: outputFile,
	})
	if err == nil || !strings.Contains(err.Error(), "output dir") {
		t.Fatalf("unexpected error: %v", err)
	}

	err = Validate(JobSpec{
		Name:    "bad-env",
		Command: "python",
		Env:     map[string]string{"BAD=KEY": "value"},
	})
	if err == nil || !strings.Contains(err.Error(), "env key") {
		t.Fatalf("unexpected env error: %v", err)
	}
}

func TestNewLoRAJobUsesDownloadedModelAndDataset(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	evalFile := filepath.Join(dir, "data", "test.jsonl")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(evalFile, []byte(`{"prompt":"bye","completion":"later"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	spec, err := NewLoRAJob(InitOptions{
		Name:      "qwen3-lora",
		BaseModel: modelDir,
		TrainFile: trainFile,
		EvalFile:  evalFile,
		OutputDir: filepath.Join(dir, "outputs", "qwen3-lora"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if spec.BaseModel != modelDir || spec.TrainFile != trainFile || spec.EvalFile != evalFile {
		t.Fatalf("unexpected spec paths: %#v", spec)
	}
	joined := strings.Join(spec.Args, " ")
	for _, want := range []string{"scripts/train_lora.py", "--model " + modelDir, "--train-file " + trainFile, "--eval-file " + evalFile} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected %q in args %q", want, joined)
		}
	}
}

func TestPreflightReportsModelAndDataset(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	evalFile := filepath.Join(dir, "data", "test.jsonl")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(evalFile, []byte(`{"prompt":"bye","completion":"later"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Preflight(JobSpec{
		Name:      "qwen3-lora",
		Method:    "lora",
		BaseModel: modelDir,
		TrainFile: trainFile,
		EvalFile:  evalFile,
		OutputDir: filepath.Join(dir, "outputs", "qwen3-lora"),
		Command:   fakeTrainingCommand(t),
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.ModelType != "qwen3" || report.TrainRows != 1 || report.EvalRows != 1 {
		t.Fatalf("unexpected report: %#v", report)
	}
	if len(report.Warnings) != 0 {
		t.Fatalf("unexpected warnings: %#v", report.Warnings)
	}
}

func TestPreflightWarnsForGGUFTrainingArtifact(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3-gguf")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.gguf"), minimalGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Preflight(JobSpec{
		Name:      "qwen3-lora",
		BaseModel: modelDir,
		TrainFile: trainFile,
		Command:   fakeTrainingCommand(t),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Warnings) == 0 || !strings.Contains(report.Warnings[0], "GGUF") {
		t.Fatalf("expected GGUF warning, got %#v", report.Warnings)
	}
}

func TestRunNativeCreatesTokenBiasAdapter(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3-gguf")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	outputDir := filepath.Join(dir, "outputs", "qwen3-adapter")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.gguf"), minimalGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0,"▁world":1,"later":2},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello world"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := RunNative(context.Background(), NativeOptions{
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		OutputDir:     outputDir,
		Epochs:        2,
		LearningRate:  0.2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.AdapterPath == "" || result.TrainRows != 1 || result.TrainTokens != 4 || result.UpdatedTokens != 2 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if _, err := os.Stat(result.AdapterPath); err != nil {
		t.Fatal(err)
	}
	if result.Adapter.Bias[0] <= 0 || result.Adapter.Bias[1] <= 0 {
		t.Fatalf("unexpected adapter bias: %#v", result.Adapter.Bias)
	}
}

func TestValidateRejectsInvalidDatasetFormat(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "model")
	trainFile := filepath.Join(dir, "train.jsonl")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"messages":[]}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := Validate(JobSpec{
		Command:       "python",
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
	})
	if err == nil || !strings.Contains(err.Error(), "completion") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestWriteJob(t *testing.T) {
	path := filepath.Join(t.TempDir(), "train-job.json")
	spec := JobSpec{Name: "job", Command: "python", Args: []string{"train.py"}}
	if err := WriteJob(path, spec); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got JobSpec
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Name != "job" || got.Command != "python" {
		t.Fatalf("unexpected job: %#v", got)
	}
}

func fakeTrainingCommand(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, "train.bat")
		if err := os.WriteFile(path, []byte("@echo off\necho training %*\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	path := filepath.Join(dir, "train.sh")
	if err := os.WriteFile(path, []byte("#!/bin/sh\necho training \"$@\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func minimalGGUF(t *testing.T) []byte {
	t.Helper()
	data := make([]byte, 4+4+8+8)
	copy(data, "GGUF")
	binary.LittleEndian.PutUint32(data[4:], 3)
	binary.LittleEndian.PutUint64(data[8:], 0)
	binary.LittleEndian.PutUint64(data[16:], 0)
	return data
}
