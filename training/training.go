package training

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/gakon/nego-ai/datasets"
	"github.com/gakon/nego-ai/modelinfo"
)

type JobSpec struct {
	Name          string            `json:"name"`
	Method        string            `json:"method,omitempty"`
	BaseModel     string            `json:"base_model,omitempty"`
	TrainFile     string            `json:"train_file,omitempty"`
	EvalFile      string            `json:"eval_file,omitempty"`
	DatasetFormat string            `json:"dataset_format,omitempty"`
	OutputDir     string            `json:"output_dir,omitempty"`
	MaxContext    int               `json:"max_context,omitempty"`
	Command       string            `json:"command"`
	Args          []string          `json:"args,omitempty"`
	Env           map[string]string `json:"env,omitempty"`
	WorkDir       string            `json:"work_dir,omitempty"`
}

type Result struct {
	Name     string        `json:"name"`
	Success  bool          `json:"success"`
	Stdout   string        `json:"stdout"`
	Stderr   string        `json:"stderr"`
	Duration time.Duration `json:"duration"`
}

type PreflightReport struct {
	Name          string              `json:"name,omitempty"`
	Method        string              `json:"method,omitempty"`
	BaseModel     string              `json:"base_model,omitempty"`
	ModelType     string              `json:"model_type,omitempty"`
	Architectures []string            `json:"architectures,omitempty"`
	TrainFile     string              `json:"train_file,omitempty"`
	TrainRows     int                 `json:"train_rows,omitempty"`
	EvalFile      string              `json:"eval_file,omitempty"`
	EvalRows      int                 `json:"eval_rows,omitempty"`
	DatasetFormat string              `json:"dataset_format,omitempty"`
	OutputDir     string              `json:"output_dir,omitempty"`
	MaxContext    int                 `json:"max_context,omitempty"`
	TrainTokens   *TokenBudgetSummary `json:"train_tokens,omitempty"`
	EvalTokens    *TokenBudgetSummary `json:"eval_tokens,omitempty"`
	Command       string              `json:"command,omitempty"`
	Args          []string            `json:"args,omitempty"`
	Warnings      []string            `json:"warnings,omitempty"`
}

type InitOptions struct {
	Name          string
	Method        string
	BaseModel     string
	TrainFile     string
	EvalFile      string
	DatasetFormat string
	OutputDir     string
	MaxContext    int
	Command       string
	Script        string
	WorkDir       string
	Env           map[string]string
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
	if opts.DatasetFormat == "" {
		opts.DatasetFormat = "auto"
	}
	spec := JobSpec{
		Name:          opts.Name,
		Method:        opts.Method,
		BaseModel:     opts.BaseModel,
		TrainFile:     opts.TrainFile,
		EvalFile:      opts.EvalFile,
		DatasetFormat: opts.DatasetFormat,
		OutputDir:     opts.OutputDir,
		MaxContext:    opts.MaxContext,
		Command:       opts.Command,
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
	if spec.MaxContext < 0 {
		return fmt.Errorf("max context must be greater than or equal to 0")
	}
	if spec.MaxContext > 0 && spec.BaseModel == "" {
		return fmt.Errorf("base model is required when max context is set")
	}
	if spec.WorkDir != "" {
		if err := requireDir("", spec.WorkDir, "work dir"); err != nil {
			return err
		}
	}
	for key := range spec.Env {
		if key == "" || strings.Contains(key, "=") {
			return fmt.Errorf("training env key %q is invalid", key)
		}
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
	if spec.OutputDir != "" {
		if err := validateOutputDir(spec.WorkDir, spec.OutputDir); err != nil {
			return err
		}
	}
	format := spec.DatasetFormat
	if format == "" {
		format = "auto"
	}
	if spec.TrainFile != "" {
		if _, err := validateDatasetFile(spec.WorkDir, spec.TrainFile, "train file", format); err != nil {
			return err
		}
	}
	if spec.EvalFile != "" {
		if _, err := validateDatasetFile(spec.WorkDir, spec.EvalFile, "eval file", format); err != nil {
			return err
		}
	}
	return nil
}

func Preflight(spec JobSpec) (PreflightReport, error) {
	report := PreflightReport{
		Name:          spec.Name,
		Method:        spec.Method,
		BaseModel:     spec.BaseModel,
		TrainFile:     spec.TrainFile,
		EvalFile:      spec.EvalFile,
		DatasetFormat: spec.DatasetFormat,
		OutputDir:     spec.OutputDir,
		MaxContext:    spec.MaxContext,
		Command:       spec.Command,
		Args:          append([]string(nil), spec.Args...),
	}
	if report.DatasetFormat == "" {
		report.DatasetFormat = "auto"
	}
	if err := Validate(spec); err != nil {
		return report, err
	}
	if spec.BaseModel != "" {
		info, err := modelinfo.Inspect(resolvePath(spec.WorkDir, spec.BaseModel))
		if err != nil {
			return report, fmt.Errorf("inspect base model: %w", err)
		}
		report.ModelType = info.ModelType
		report.Architectures = append([]string(nil), info.Architectures...)
		report.Warnings = append(report.Warnings, trainingModelWarnings(info)...)
	}
	if spec.TrainFile != "" {
		rows, err := validateDatasetFile(spec.WorkDir, spec.TrainFile, "train file", report.DatasetFormat)
		if err != nil {
			return report, err
		}
		report.TrainRows = rows
		if spec.MaxContext > 0 {
			budget, err := analyzeDatasetTokenBudget(spec.WorkDir, spec.TrainFile, "train file", report.DatasetFormat, spec.BaseModel, spec.MaxContext)
			report.TrainTokens = budget
			if err != nil {
				return report, err
			}
		}
	}
	if spec.EvalFile != "" {
		rows, err := validateDatasetFile(spec.WorkDir, spec.EvalFile, "eval file", report.DatasetFormat)
		if err != nil {
			return report, err
		}
		report.EvalRows = rows
		if spec.MaxContext > 0 {
			budget, err := analyzeDatasetTokenBudget(spec.WorkDir, spec.EvalFile, "eval file", report.DatasetFormat, spec.BaseModel, spec.MaxContext)
			report.EvalTokens = budget
			if err != nil {
				return report, err
			}
		}
	}
	return report, nil
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
	if _, err := Preflight(spec); err != nil {
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

func requireDir(workDir, path, label string) error {
	if err := requirePath(workDir, path, label); err != nil {
		return err
	}
	resolved := resolvePath(workDir, path)
	info, err := os.Stat(resolved)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("%s %q must be a directory", label, path)
	}
	return nil
}

func validateOutputDir(workDir, path string) error {
	resolved := resolvePath(workDir, path)
	info, err := os.Stat(resolved)
	if err == nil {
		if !info.IsDir() {
			return fmt.Errorf("output dir %q must be a directory", path)
		}
		return nil
	}
	if os.IsNotExist(err) {
		return nil
	}
	return fmt.Errorf("output dir %q is not available: %w", path, err)
}

func validateDatasetFile(workDir, path, label, format string) (int, error) {
	rows, err := datasets.ReadFile(resolvePath(workDir, path))
	if err != nil {
		return 0, fmt.Errorf("%s %q is invalid: %w", label, path, err)
	}
	if len(rows) == 0 {
		return 0, fmt.Errorf("%s %q has no rows", label, path)
	}
	if err := datasets.ValidateFormat(rows, format); err != nil {
		return 0, fmt.Errorf("%s %q does not match %s format: %w", label, path, format, err)
	}
	return len(rows), nil
}

type TokenBudgetSummary struct {
	Rows          int                       `json:"rows"`
	CountedRows   int                       `json:"counted_rows"`
	MaxContext    int                       `json:"max_context"`
	TotalTokens   int                       `json:"total_tokens"`
	MinTokens     int                       `json:"min_tokens"`
	MaxTokens     int                       `json:"max_tokens"`
	AverageTokens float64                   `json:"average_tokens"`
	OverLimit     int                       `json:"over_limit"`
	InvalidRows   int                       `json:"invalid_rows"`
	LongestRows   []datasets.TokenBudgetRow `json:"longest_rows,omitempty"`
}

func analyzeDatasetTokenBudget(workDir, path, label, format, baseModel string, maxContext int) (*TokenBudgetSummary, error) {
	rows, err := datasets.ReadFile(resolvePath(workDir, path))
	if err != nil {
		return nil, fmt.Errorf("%s %q is invalid: %w", label, path, err)
	}
	report, err := datasets.AnalyzeTokenBudget(rows, datasets.TokenBudgetOptions{
		ModelPath:  resolvePath(workDir, baseModel),
		Format:     format,
		MaxContext: maxContext,
	})
	if err != nil {
		return nil, fmt.Errorf("%s %q token budget failed: %w", label, path, err)
	}
	summary := summarizeTokenBudget(report)
	if !report.Valid {
		return summary, fmt.Errorf("%s %q exceeds token budget: over_limit=%d invalid_rows=%d max_context=%d", label, path, summary.OverLimit, summary.InvalidRows, maxContext)
	}
	return summary, nil
}

func summarizeTokenBudget(report datasets.TokenBudgetReport) *TokenBudgetSummary {
	summary := &TokenBudgetSummary{
		Rows:          report.Rows,
		CountedRows:   report.CountedRows,
		MaxContext:    report.MaxContext,
		TotalTokens:   report.TotalTokens,
		MinTokens:     report.MinTokens,
		MaxTokens:     report.MaxTokens,
		AverageTokens: report.AverageTokens,
		OverLimit:     report.OverLimit,
	}
	for _, row := range report.RowsDetail {
		if row.Error != "" {
			summary.InvalidRows++
		}
	}
	summary.LongestRows = datasets.LongestTokenRows(report.RowsDetail, 5)
	return summary
}

func trainingModelWarnings(info *modelinfo.Info) []string {
	if info == nil {
		return nil
	}
	hasSafetensors := false
	hasGGUF := false
	hasTokenizer := false
	for _, file := range info.Files {
		switch file.Kind {
		case "safetensors":
			hasSafetensors = true
		case "gguf":
			hasGGUF = true
		case "tokenizer", "tokenizer_config":
			hasTokenizer = true
		}
	}
	var warnings []string
	if hasGGUF && !hasSafetensors {
		warnings = append(warnings, "base model looks like a GGUF runtime artifact; most fine-tuning tools expect a Hugging Face model directory with safetensors")
	}
	if !hasSafetensors && info.GGUF == nil {
		warnings = append(warnings, "base model has no safetensors or GGUF weights detected")
	}
	if !hasTokenizer {
		warnings = append(warnings, "base model has no tokenizer.json or tokenizer_config.json detected")
	}
	return warnings
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
