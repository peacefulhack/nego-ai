package training

import (
	"context"
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
