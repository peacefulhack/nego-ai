package training

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/datasets"
	"github.com/gakon/nego-ai/modelinfo"
	"github.com/gakon/nego-ai/tokenizer"
)

type NativeOptions struct {
	BaseModel     string  `json:"base_model"`
	TrainFile     string  `json:"train_file"`
	EvalFile      string  `json:"eval_file,omitempty"`
	DatasetFormat string  `json:"dataset_format,omitempty"`
	OutputDir     string  `json:"output_dir"`
	Method        string  `json:"method,omitempty"`
	LearningRate  float64 `json:"learning_rate,omitempty"`
	Epochs        int     `json:"epochs,omitempty"`
}

type NativeResult struct {
	BaseModel     string                     `json:"base_model"`
	TrainFile     string                     `json:"train_file"`
	EvalFile      string                     `json:"eval_file,omitempty"`
	DatasetFormat string                     `json:"dataset_format"`
	OutputDir     string                     `json:"output_dir"`
	AdapterPath   string                     `json:"adapter_path"`
	Method        string                     `json:"method"`
	Epochs        int                        `json:"epochs"`
	TrainRows     int                        `json:"train_rows"`
	EvalRows      int                        `json:"eval_rows,omitempty"`
	TrainTokens   int                        `json:"train_tokens"`
	VocabSize     int                        `json:"vocab_size"`
	UpdatedTokens int                        `json:"updated_tokens"`
	Duration      time.Duration              `json:"duration"`
	Artifact      *modelinfo.Artifact        `json:"artifact,omitempty"`
	Adapter       *adapters.TokenBiasAdapter `json:"adapter,omitempty"`
	Warnings      []string                   `json:"warnings,omitempty"`
}

func RunNative(ctx context.Context, opts NativeOptions) (NativeResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()
	result := NativeResult{
		BaseModel:     opts.BaseModel,
		TrainFile:     opts.TrainFile,
		EvalFile:      opts.EvalFile,
		DatasetFormat: opts.DatasetFormat,
		OutputDir:     opts.OutputDir,
		Method:        opts.Method,
		Epochs:        opts.Epochs,
	}
	normalized, err := normalizeNativeOptions(opts)
	if err != nil {
		return result, err
	}
	result.DatasetFormat = normalized.DatasetFormat
	result.OutputDir = normalized.OutputDir
	result.Method = normalized.Method
	result.Epochs = normalized.Epochs
	artifact, err := modelinfo.Resolve(normalized.BaseModel)
	if err != nil {
		return result, fmt.Errorf("resolve base model: %w", err)
	}
	result.Artifact = artifact
	if artifact.Format != modelinfo.ArtifactFormatGGUF && artifact.Format != modelinfo.ArtifactFormatMixed {
		return result, fmt.Errorf("native GGUF training requires a GGUF model artifact, got %s", artifact.Format)
	}
	tok, err := loadNativeTrainingTokenizer(normalized.BaseModel, artifact)
	if err != nil {
		return result, err
	}
	result.VocabSize = tok.vocabSize()
	rows, err := datasets.ReadFile(resolvePath("", normalized.TrainFile))
	if err != nil {
		return result, fmt.Errorf("train file %q is invalid: %w", normalized.TrainFile, err)
	}
	if len(rows) == 0 {
		return result, fmt.Errorf("train file %q has no rows", normalized.TrainFile)
	}
	if err := datasets.ValidateFormat(rows, normalized.DatasetFormat); err != nil {
		return result, fmt.Errorf("train file %q does not match %s format: %w", normalized.TrainFile, normalized.DatasetFormat, err)
	}
	result.TrainRows = len(rows)
	if normalized.EvalFile != "" {
		evalRows, err := datasets.ReadFile(resolvePath("", normalized.EvalFile))
		if err != nil {
			return result, fmt.Errorf("eval file %q is invalid: %w", normalized.EvalFile, err)
		}
		if err := datasets.ValidateFormat(evalRows, normalized.DatasetFormat); err != nil {
			return result, fmt.Errorf("eval file %q does not match %s format: %w", normalized.EvalFile, normalized.DatasetFormat, err)
		}
		result.EvalRows = len(evalRows)
	}
	counts := make(map[int]int)
	for epoch := 0; epoch < normalized.Epochs; epoch++ {
		for _, text := range targetTexts(rows, normalized.DatasetFormat) {
			if err := ctx.Err(); err != nil {
				return result, err
			}
			ids, err := tok.encode(text)
			if err != nil {
				return result, err
			}
			result.TrainTokens += len(ids)
			for _, id := range ids {
				counts[id]++
			}
		}
	}
	if len(counts) == 0 {
		return result, fmt.Errorf("training dataset produced no target tokens")
	}
	adapter := adapters.NewTokenBias(normalized.BaseModel, normalized.Method, tok.vocabSize(), counts, normalized.LearningRate)
	adapterPath := filepath.Join(normalized.OutputDir, "adapter.json")
	if err := adapters.Save(adapterPath, adapter); err != nil {
		return result, err
	}
	result.AdapterPath = adapterPath
	result.Adapter = adapter
	result.UpdatedTokens = len(adapter.Bias)
	result.Duration = time.Since(start)
	result.Warnings = []string{
		"native GGUF training currently writes a token-bias adapter; full LoRA/backprop training is still planned",
		"load the adapter with native backend option adapter_path",
	}
	return result, nil
}

func normalizeNativeOptions(opts NativeOptions) (NativeOptions, error) {
	opts.BaseModel = strings.TrimSpace(opts.BaseModel)
	opts.TrainFile = strings.TrimSpace(opts.TrainFile)
	opts.EvalFile = strings.TrimSpace(opts.EvalFile)
	opts.OutputDir = strings.TrimSpace(opts.OutputDir)
	opts.DatasetFormat = strings.ToLower(strings.TrimSpace(opts.DatasetFormat))
	opts.Method = strings.ToLower(strings.TrimSpace(opts.Method))
	if opts.BaseModel == "" {
		return opts, fmt.Errorf("base model is required")
	}
	if opts.TrainFile == "" {
		return opts, fmt.Errorf("train file is required")
	}
	if opts.OutputDir == "" {
		return opts, fmt.Errorf("output dir is required")
	}
	if opts.DatasetFormat == "" {
		opts.DatasetFormat = "auto"
	}
	if opts.Method == "" {
		opts.Method = "token-bias"
	}
	if opts.Method != "token-bias" {
		return opts, fmt.Errorf("unsupported native training method %q", opts.Method)
	}
	if opts.Epochs <= 0 {
		opts.Epochs = 1
	}
	if opts.LearningRate <= 0 {
		opts.LearningRate = 0.1
	}
	if _, err := os.Stat(opts.BaseModel); err != nil {
		return opts, fmt.Errorf("base model %q is not available: %w", opts.BaseModel, err)
	}
	return opts, nil
}

type nativeTrainingTokenizer interface {
	encode(string) ([]int, error)
	vocabSize() int
}

type jsonTrainingTokenizer struct {
	tok *tokenizer.Tokenizer
}

func (t jsonTrainingTokenizer) encode(text string) ([]int, error) {
	return t.tok.Encode(text)
}

func (t jsonTrainingTokenizer) vocabSize() int {
	return len(t.tok.IDToToken)
}

type ggufTrainingTokenizer struct {
	vocab *modelinfo.GGUFVocab
}

func (t ggufTrainingTokenizer) encode(text string) ([]int, error) {
	return t.vocab.Encode(text, modelinfo.EncodeOptions{})
}

func (t ggufTrainingTokenizer) vocabSize() int {
	return len(t.vocab.Tokens)
}

func loadNativeTrainingTokenizer(baseModel string, artifact *modelinfo.Artifact) (nativeTrainingTokenizer, error) {
	if tok, err := tokenizer.Load(baseModel); err == nil {
		return jsonTrainingTokenizer{tok: tok}, nil
	}
	if artifact != nil && artifact.RuntimeFile != nil && artifact.RuntimeFile.Kind == "gguf" {
		vocab, err := modelinfo.InspectGGUFVocab(artifact.RuntimeFile.Path)
		if err != nil {
			return nil, fmt.Errorf("load GGUF tokenizer: %w", err)
		}
		if len(vocab.Tokens) == 0 {
			return nil, fmt.Errorf("GGUF tokenizer has no tokens")
		}
		return ggufTrainingTokenizer{vocab: vocab}, nil
	}
	return nil, fmt.Errorf("native training requires tokenizer.json beside the model or GGUF tokenizer metadata")
}

func targetTexts(rows []datasets.Row, format string) []string {
	texts := make([]string, 0, len(rows))
	for _, row := range rows {
		text := targetText(row, format)
		if strings.TrimSpace(text) != "" {
			texts = append(texts, text)
		}
	}
	return texts
}

func targetText(row datasets.Row, format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "completion":
		return rowString(row, "completion", "response")
	case "instruction":
		return rowString(row, "output")
	case "chat":
		return lastAssistantMessage(row)
	default:
		if _, ok := row["messages"]; ok {
			return lastAssistantMessage(row)
		}
		if _, ok := row["completion"]; ok {
			return rowString(row, "completion")
		}
		if _, ok := row["response"]; ok {
			return rowString(row, "response")
		}
		return rowString(row, "output")
	}
}

func rowString(row datasets.Row, keys ...string) string {
	for _, key := range keys {
		if value, ok := row[key]; ok {
			return strings.TrimSpace(fmt.Sprint(value))
		}
	}
	return ""
}

func lastAssistantMessage(row datasets.Row) string {
	raw, ok := row["messages"].([]any)
	if !ok {
		return ""
	}
	for i := len(raw) - 1; i >= 0; i-- {
		message, ok := raw[i].(map[string]any)
		if !ok {
			continue
		}
		if role, _ := message["role"].(string); role != "assistant" {
			continue
		}
		content, _ := message["content"].(string)
		return strings.TrimSpace(content)
	}
	return ""
}
