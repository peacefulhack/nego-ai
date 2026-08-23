package training

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

type JobSpec struct {
	Name      string            `json:"name"`
	Method    string            `json:"method,omitempty"`
	BaseModel string            `json:"base_model,omitempty"`
	TrainFile string            `json:"train_file,omitempty"`
	EvalFile  string            `json:"eval_file,omitempty"`
	OutputDir string            `json:"output_dir,omitempty"`
	Command   string            `json:"command"`
	Args      []string          `json:"args,omitempty"`
	Env       map[string]string `json:"env,omitempty"`
	WorkDir   string            `json:"work_dir,omitempty"`
}

type Result struct {
	Name     string        `json:"name"`
	Success  bool          `json:"success"`
	Stdout   string        `json:"stdout"`
	Stderr   string        `json:"stderr"`
	Duration time.Duration `json:"duration"`
}

type InitOptions struct {
	Name      string
	Method    string
	BaseModel string
	TrainFile string
	EvalFile  string
	OutputDir string
	Command   string
	Script    string
	WorkDir   string
	Env       map[string]string
}

func NewLoRAJob(opts InitOptions) (JobSpec, error) {
	if opts.Name == "" {
		opts.Name = "nego-lora"
	}
	if opts.Method == "" {
		opts.Method = "lora"
	}
	if opts.Command == "" {
		opts.Command = "python"
	}
	if opts.Script == "" {
		opts.Script = "scripts/train_lora.py"
	}
	if opts.OutputDir == "" {
		opts.OutputDir = "./outputs/" + opts.Name
	}
	spec := JobSpec{
		Name:      opts.Name,
		Method:    opts.Method,
		BaseModel: opts.BaseModel,
		TrainFile: opts.TrainFile,
		EvalFile:  opts.EvalFile,
		OutputDir: opts.OutputDir,
		Command:   opts.Command,
		Args: []string{
			opts.Script,
			"--model", opts.BaseModel,
			"--train-file", opts.TrainFile,
			"--output-dir", opts.OutputDir,
		},
		Env:     copyStringMap(opts.Env),
		WorkDir: opts.WorkDir,
	}
	if spec.Env == nil {
		spec.Env = map[string]string{"TOKENIZERS_PARALLELISM": "false"}
	}
	if opts.EvalFile != "" {
		spec.Args = append(spec.Args, "--eval-file", opts.EvalFile)
	}
	return spec, Validate(spec)
}

func Validate(spec JobSpec) error {
	if spec.Command == "" {
		return fmt.Errorf("training command is required")
	}
	if spec.BaseModel != "" {
		if err := requirePath(spec.WorkDir, spec.BaseModel, "base model"); err != nil {
			return err
		}
	}
	if spec.TrainFile != "" {
		if err := requireFile(spec.WorkDir, spec.TrainFile, "train file"); err != nil {
			return err
		}
	}
	if spec.EvalFile != "" {
		if err := requireFile(spec.WorkDir, spec.EvalFile, "eval file"); err != nil {
			return err
		}
	}
	return nil
}

func WriteJob(path string, spec JobSpec) error {
	if path == "" {
		return fmt.Errorf("output path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func Run(ctx context.Context, spec JobSpec) (Result, error) {
	if err := Validate(spec); err != nil {
		return Result{Name: spec.Name}, err
	}
	start := time.Now()
	cmd := exec.CommandContext(ctx, spec.Command, spec.Args...)
	cmd.Dir = spec.WorkDir
	cmd.Env = os.Environ()
	for key, value := range spec.Env {
		cmd.Env = append(cmd.Env, key+"="+value)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	result := Result{
		Name:     spec.Name,
		Success:  err == nil,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: time.Since(start),
	}
	if err != nil {
		return result, err
	}
	return result, nil
}

func requirePath(workDir, path, label string) error {
	if path == "" {
		return fmt.Errorf("%s is required", label)
	}
	resolved := resolvePath(workDir, path)
	if _, err := os.Stat(resolved); err != nil {
		return fmt.Errorf("%s %q is not available: %w", label, path, err)
	}
	return nil
}

func requireFile(workDir, path, label string) error {
	if err := requirePath(workDir, path, label); err != nil {
		return err
	}
	resolved := resolvePath(workDir, path)
	info, err := os.Stat(resolved)
	if err != nil {
		return err
	}
	if info.IsDir() {
		return fmt.Errorf("%s %q must be a file", label, path)
	}
	return nil
}

func resolvePath(workDir, path string) string {
	if filepath.IsAbs(path) || workDir == "" {
		return path
	}
	return filepath.Join(workDir, path)
}

func copyStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}
