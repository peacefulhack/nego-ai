package training

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
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

func TestPreflightChecksTokenBudget(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","unk_token":"[UNK]","vocab":{"[UNK]":0,"hello":1,"world":2,"Ġhello":3,"Ġworld":4,"Ċ":5}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hello hello hello","completion":"world"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Preflight(JobSpec{
		Name:          "qwen3-lora",
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		MaxContext:    3,
		Command:       fakeTrainingCommand(t),
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds token budget") {
		t.Fatalf("expected token budget error, got report=%#v err=%v", report, err)
	}
	if report.TrainTokens == nil || report.TrainTokens.OverLimit != 1 || report.TrainTokens.MaxTokens <= 3 {
		t.Fatalf("unexpected token summary: %#v", report.TrainTokens)
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
		MaxContext:    8,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.AdapterPath == "" || result.TrainRows != 1 || result.TrainTokens != 4 || result.UpdatedTokens != 2 || result.TrainBudget == nil || len(result.TopTokens) != 2 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if result.Memory == nil || result.Memory.TotalBytes == 0 {
		t.Fatalf("expected base memory estimate: %#v", result.Memory)
	}
	if result.TopTokens[0].Text != "hello" || result.TopTokens[0].Count != 2 {
		t.Fatalf("unexpected top tokens: %#v", result.TopTokens)
	}
	if _, err := os.Stat(result.AdapterPath); err != nil {
		t.Fatal(err)
	}
	if result.ManifestPath == "" || result.ReadmePath == "" {
		t.Fatalf("missing output metadata: %#v", result)
	}
	data, err := os.ReadFile(result.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var manifest NativeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Type != "nego-native-adapter" || manifest.RecommendedBackend != "native" || manifest.AdapterPath != "adapter.json" || manifest.RuntimeOptions["adapter_path"] != "adapter.json" || len(manifest.TopTokens) != 2 || manifest.BaseMemory == nil {
		t.Fatalf("unexpected manifest: %#v", manifest)
	}
	if manifest.LearningRate != 0.2 || manifest.Epochs != 2 || manifest.MaxContext != 8 || manifest.TrainRows != 1 || manifest.BaseFormat != "gguf" {
		t.Fatalf("unexpected manifest: %#v", manifest)
	}
	if !warningsContain(manifest.Warnings, "native generation is not ready yet") {
		t.Fatalf("expected manifest runtime warning: %#v", manifest.Warnings)
	}
	loaded, err := LoadNativeManifest(outputDir)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.AdapterPath != "adapter.json" {
		t.Fatalf("loaded manifest = %#v", loaded)
	}
	loaded, err = LoadNativeManifest(result.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.BaseModel != modelDir {
		t.Fatalf("loaded manifest = %#v", loaded)
	}
	readme, err := os.ReadFile(result.ReadmePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(readme), "nego run "+outputDir) || !strings.Contains(string(readme), "nego chat "+outputDir) {
		t.Fatalf("unexpected README: %s", string(readme))
	}
	if !strings.Contains(string(readme), "Base memory estimate:") {
		t.Fatalf("expected README memory estimate: %s", string(readme))
	}
	if !strings.Contains(string(readme), "Warnings:") || !strings.Contains(string(readme), "native generation is not ready yet") {
		t.Fatalf("expected README warning: %s", string(readme))
	}
	if result.Adapter.Bias[0] <= 0 || result.Adapter.Bias[1] <= 0 {
		t.Fatalf("unexpected adapter bias: %#v", result.Adapter.Bias)
	}
	if !warningsContain(result.Warnings, "native generation is not ready yet") {
		t.Fatalf("expected runtime readiness warning: %#v", result.Warnings)
	}
}

func TestRunNativeManifestPersistsGGUFTokenizer(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "models", "model.gguf")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	outputDir := filepath.Join(dir, "outputs", "qwen3-adapter")
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath, tokenizerGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result, err := RunNative(context.Background(), NativeOptions{
		BaseModel:     modelPath,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		OutputDir:     outputDir,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Tokenizer == nil || result.Tokenizer.Format != "gguf" || result.Tokenizer.Model != "llama" || result.Tokenizer.PreTokenizer != "llama-bpe" {
		t.Fatalf("unexpected result tokenizer: %#v", result.Tokenizer)
	}
	data, err := os.ReadFile(result.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var manifest NativeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Tokenizer == nil || manifest.Tokenizer.VocabSize != 2 || manifest.Tokenizer.PreTokenizer != "llama-bpe" {
		t.Fatalf("unexpected manifest tokenizer: %#v", manifest.Tokenizer)
	}
	readme, err := os.ReadFile(result.ReadmePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(readme), "Tokenizer: `gguf` `llama` pre-tokenizer `llama-bpe` vocab `2`") {
		t.Fatalf("expected README tokenizer summary: %s", string(readme))
	}
}

func TestRunNativeDryRunDoesNotWriteAdapter(t *testing.T) {
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
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0,"▁world":1},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello world"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stages []string
	result, err := RunNative(context.Background(), NativeOptions{
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		OutputDir:     outputDir,
		DryRun:        true,
		MaxContext:    8,
		Progress: func(event NativeProgress) {
			stages = append(stages, event.Stage)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.DryRun || result.AdapterPath == "" || result.UpdatedTokens != 2 || result.TrainBudget == nil || len(result.TopTokens) != 2 {
		t.Fatalf("unexpected dry run result: %#v", result)
	}
	if _, err := os.Stat(filepath.Join(outputDir, "adapter.json")); !os.IsNotExist(err) {
		t.Fatalf("adapter should not be written during dry run: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outputDir, "manifest.json")); !os.IsNotExist(err) {
		t.Fatalf("manifest should not be written during dry run: %v", err)
	}
	for _, want := range []string{"resolve", "tokenizer", "read_train", "train_budget", "train", "done"} {
		if !containsStage(stages, want) {
			t.Fatalf("expected progress stage %q in %#v", want, stages)
		}
	}

	result, err = RunNative(context.Background(), NativeOptions{
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		DryRun:        true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.AdapterPath != "" || result.OutputDir != "" {
		t.Fatalf("dry run without output should not plan files: %#v", result)
	}
}

func containsStage(stages []string, want string) bool {
	for _, stage := range stages {
		if stage == want {
			return true
		}
	}
	return false
}

func warningsContain(warnings []string, needle string) bool {
	for _, warning := range warnings {
		if strings.Contains(warning, needle) {
			return true
		}
	}
	return false
}

func TestNativeManifestRejectsEscapingAdapterPath(t *testing.T) {
	manifest := NativeManifest{
		Version:            1,
		Type:               "nego-native-adapter",
		BaseModel:          "./models/qwen3",
		AdapterPath:        "../adapter.json",
		RecommendedBackend: "native",
	}
	if err := manifest.Validate(); err == nil || !strings.Contains(err.Error(), "cannot escape") {
		t.Fatalf("expected escaping adapter path error, got %v", err)
	}

	manifest.AdapterPath = "adapter.json"
	manifest.RuntimeOptions = map[string]string{"adapter_path": "../adapter.json"}
	if err := manifest.Validate(); err == nil || !strings.Contains(err.Error(), "cannot escape") {
		t.Fatalf("expected escaping runtime option error, got %v", err)
	}
}

func TestNativeBaseRuntimeWarningsIncludeCompatibleBackendWarnings(t *testing.T) {
	report := &modelinfo.CheckReport{
		Backends: []modelinfo.BackendCompatibility{{
			Name:       "native",
			Compatible: true,
			Warnings:   []string{"tokenizer output should be validated"},
		}},
	}
	artifact := &modelinfo.Artifact{Format: modelinfo.ArtifactFormatGGUF}
	warnings := nativeBaseRuntimeWarnings(report, artifact)
	if !warningsContain(warnings, "tokenizer output should be validated") {
		t.Fatalf("expected backend warning: %#v", warnings)
	}
}

func TestRunNativeRejectsRowsOverTokenBudget(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	outputDir := filepath.Join(dir, "outputs", "qwen3-adapter")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"num_hidden_layers":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0,"▁hello":1,"world":2,"Ċ":3},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), minimalSafetensors(t), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hello hello hello","completion":"world"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result, err := RunNative(context.Background(), NativeOptions{
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		OutputDir:     outputDir,
		MaxContext:    2,
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds token budget") {
		t.Fatalf("expected token budget error, got result=%#v err=%v", result, err)
	}
	if result.TrainBudget == nil || result.TrainBudget.OverLimit != 1 {
		t.Fatalf("unexpected budget: %#v", result.TrainBudget)
	}
	if _, err := os.Stat(filepath.Join(outputDir, "adapter.json")); !os.IsNotExist(err) {
		t.Fatalf("adapter should not be written when budget fails: %v", err)
	}
}

func TestRunNativeSupportsHFSafetensorsDirectory(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	outputDir := filepath.Join(dir, "outputs", "qwen3-adapter")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"num_hidden_layers":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0,"▁world":1},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), minimalSafetensors(t), 0o644); err != nil {
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
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Artifact == nil || string(result.Artifact.Format) != "hf-safetensors" || result.UpdatedTokens != 2 {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestRunNativeDuplicateRows(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	outputDir := filepath.Join(dir, "outputs", "qwen3-adapter")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"num_hidden_layers":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0,"▁world":1},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), minimalSafetensors(t), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"prompt":"hi","completion":"hello world"}` + "\n" +
		`{"prompt":" hi ","completion":"hello world"}` + "\n"
	if err := os.WriteFile(trainFile, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	result, err := RunNative(context.Background(), NativeOptions{
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		OutputDir:     outputDir,
		DryRun:        true,
		DedupeKeys:    []string{"prompt", "completion"},
		DedupeTrim:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.DuplicateRows != 1 || len(result.Warnings) == 0 || !strings.Contains(result.Warnings[0], "duplicate") {
		t.Fatalf("unexpected result: %#v", result)
	}

	result, err = RunNative(context.Background(), NativeOptions{
		BaseModel:     modelDir,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		OutputDir:     outputDir,
		DedupeKeys:    []string{"prompt", "completion"},
		DedupeTrim:    true,
		FailOnDupes:   true,
	})
	if err == nil || !strings.Contains(err.Error(), "duplicate rows") {
		t.Fatalf("expected duplicate error, got result=%#v err=%v", result, err)
	}
	if _, err := os.Stat(filepath.Join(outputDir, "adapter.json")); !os.IsNotExist(err) {
		t.Fatalf("adapter should not be written when duplicate check fails: %v", err)
	}
}

func TestAssessReportsTrainingCapabilities(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"num_hidden_layers":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), minimalSafetensors(t), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Assess(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	if !methodAvailable(report.Methods, "token-bias") {
		t.Fatalf("expected token-bias support: %#v", report.Methods)
	}
	if !methodStatus(report.Methods, "native-lora", MethodPlanned) {
		t.Fatalf("expected planned native-lora support: %#v", report.Methods)
	}
	if report.Memory == nil || report.Memory.TotalBytes == 0 {
		t.Fatalf("expected memory estimate: %#v", report.Memory)
	}
}

func TestAssessReportsGGUFTokenizer(t *testing.T) {
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, tokenizerGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Assess(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if report.Tokenizer == nil ||
		report.Tokenizer.Format != "gguf" ||
		report.Tokenizer.Model != "llama" ||
		report.Tokenizer.VocabSize != 2 {
		t.Fatalf("unexpected tokenizer report: %#v", report.Tokenizer)
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

func methodAvailable(methods []MethodSupport, name string) bool {
	for _, method := range methods {
		if method.Method == name {
			return method.Available
		}
	}
	return false
}

func methodStatus(methods []MethodSupport, name string, status MethodStatus) bool {
	for _, method := range methods {
		if method.Method == name {
			return method.Status == status
		}
	}
	return false
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

func tokenizerGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	for _, value := range []any{uint32(3), uint64(0), uint64(3)} {
		if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeTrainingGGUFStringKV(t, &buf, "tokenizer.ggml.model", "llama")
	writeTrainingGGUFStringKV(t, &buf, "tokenizer.ggml.pre", "llama-bpe")
	writeTrainingGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<s>", "hello"})
	return buf.Bytes()
}

func writeTrainingGGUFStringKV(t *testing.T, buf *bytes.Buffer, key, value string) {
	t.Helper()
	writeTrainingGGUFString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(8)); err != nil {
		t.Fatal(err)
	}
	writeTrainingGGUFString(t, buf, value)
}

func writeTrainingGGUFStringArrayKV(t *testing.T, buf *bytes.Buffer, key string, values []string) {
	t.Helper()
	writeTrainingGGUFString(t, buf, key)
	for _, value := range []any{uint32(9), uint32(8), uint64(len(values))} {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range values {
		writeTrainingGGUFString(t, buf, value)
	}
}

func writeTrainingGGUFString(t *testing.T, buf *bytes.Buffer, value string) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	buf.WriteString(value)
}

func minimalSafetensors(t *testing.T) []byte {
	t.Helper()
	header := `{"model.embed_tokens.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	data := make([]byte, 8+len(header)+4)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	return data
}
