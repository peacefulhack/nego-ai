package training

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/datasets"
	"github.com/gakon/nego-ai/modelinfo"
	"github.com/gakon/nego-ai/tokenizer"
)

const maxNativeManifestBytes = 1 << 20

type NativeOptions struct {
	BaseModel     string  `json:"base_model"`
	TrainFile     string  `json:"train_file"`
	EvalFile      string  `json:"eval_file,omitempty"`
	DatasetFormat string  `json:"dataset_format,omitempty"`
	OutputDir     string  `json:"output_dir"`
	Method        string  `json:"method,omitempty"`
	LearningRate  float64 `json:"learning_rate,omitempty"`
	Epochs        int     `json:"epochs,omitempty"`
	MaxContext    int     `json:"max_context,omitempty"`
}

type NativeResult struct {
	BaseModel     string                     `json:"base_model"`
	TrainFile     string                     `json:"train_file"`
	EvalFile      string                     `json:"eval_file,omitempty"`
	DatasetFormat string                     `json:"dataset_format"`
	OutputDir     string                     `json:"output_dir"`
	AdapterPath   string                     `json:"adapter_path"`
	ManifestPath  string                     `json:"manifest_path,omitempty"`
	ReadmePath    string                     `json:"readme_path,omitempty"`
	Method        string                     `json:"method"`
	Epochs        int                        `json:"epochs"`
	MaxContext    int                        `json:"max_context,omitempty"`
	TrainRows     int                        `json:"train_rows"`
	EvalRows      int                        `json:"eval_rows,omitempty"`
	TrainBudget   *TokenBudgetSummary        `json:"train_budget,omitempty"`
	EvalBudget    *TokenBudgetSummary        `json:"eval_budget,omitempty"`
	TrainTokens   int                        `json:"train_tokens"`
	VocabSize     int                        `json:"vocab_size"`
	UpdatedTokens int                        `json:"updated_tokens"`
	Duration      time.Duration              `json:"duration"`
	Artifact      *modelinfo.Artifact        `json:"artifact,omitempty"`
	Adapter       *adapters.TokenBiasAdapter `json:"adapter,omitempty"`
	Warnings      []string                   `json:"warnings,omitempty"`
}

type NativeManifest struct {
	Version            int               `json:"version"`
	Type               string            `json:"type"`
	BaseModel          string            `json:"base_model"`
	AdapterPath        string            `json:"adapter_path"`
	Method             string            `json:"method"`
	DatasetFormat      string            `json:"dataset_format"`
	TrainFile          string            `json:"train_file"`
	EvalFile           string            `json:"eval_file,omitempty"`
	RecommendedBackend string            `json:"recommended_backend"`
	RuntimeOptions     map[string]string `json:"runtime_options"`
	RunArgs            []string          `json:"run_args"`
	ChatArgs           []string          `json:"chat_args"`
	VocabSize          int               `json:"vocab_size"`
	UpdatedTokens      int               `json:"updated_tokens"`
	TrainTokens        int               `json:"train_tokens"`
	CreatedAt          time.Time         `json:"created_at"`
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
		MaxContext:    opts.MaxContext,
	}
	normalized, err := normalizeNativeOptions(opts)
	if err != nil {
		return result, err
	}
	result.DatasetFormat = normalized.DatasetFormat
	result.OutputDir = normalized.OutputDir
	result.Method = normalized.Method
	result.Epochs = normalized.Epochs
	result.MaxContext = normalized.MaxContext
	artifact, err := modelinfo.Resolve(normalized.BaseModel)
	if err != nil {
		return result, fmt.Errorf("resolve base model: %w", err)
	}
	result.Artifact = artifact
	if !nativeTrainingFormat(artifact.Format) {
		return result, fmt.Errorf("native token-bias training requires GGUF or Hugging Face safetensors weights, got %s", artifact.Format)
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
	if normalized.MaxContext > 0 {
		budget, err := nativeRowsTokenBudget(rows, normalized.DatasetFormat, tok, normalized.MaxContext)
		result.TrainBudget = budget
		if err != nil {
			return result, fmt.Errorf("train file %q exceeds token budget: %w", normalized.TrainFile, err)
		}
	}
	if normalized.EvalFile != "" {
		evalRows, err := datasets.ReadFile(resolvePath("", normalized.EvalFile))
		if err != nil {
			return result, fmt.Errorf("eval file %q is invalid: %w", normalized.EvalFile, err)
		}
		if err := datasets.ValidateFormat(evalRows, normalized.DatasetFormat); err != nil {
			return result, fmt.Errorf("eval file %q does not match %s format: %w", normalized.EvalFile, normalized.DatasetFormat, err)
		}
		result.EvalRows = len(evalRows)
		if normalized.MaxContext > 0 {
			budget, err := nativeRowsTokenBudget(evalRows, normalized.DatasetFormat, tok, normalized.MaxContext)
			result.EvalBudget = budget
			if err != nil {
				return result, fmt.Errorf("eval file %q exceeds token budget: %w", normalized.EvalFile, err)
			}
		}
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
	manifest, err := buildNativeManifest(normalized, result, artifact)
	if err != nil {
		return result, err
	}
	manifestPath := filepath.Join(normalized.OutputDir, "manifest.json")
	if err := writeNativeManifest(manifestPath, manifest); err != nil {
		return result, err
	}
	result.ManifestPath = manifestPath
	readmePath := filepath.Join(normalized.OutputDir, "README.md")
	if err := writeNativeTrainingReadme(readmePath, manifest); err != nil {
		return result, err
	}
	result.ReadmePath = readmePath
	result.Duration = time.Since(start)
	result.Warnings = []string{
		"native training currently writes a token-bias adapter; full LoRA/backprop training is still planned",
		"load the adapter with native backend option adapter_path when using a compatible runtime vocabulary",
	}
	return result, nil
}

func buildNativeManifest(opts NativeOptions, result NativeResult, artifact *modelinfo.Artifact) (NativeManifest, error) {
	backend := nativeTrainingRuntimeBackend(artifact)
	if backend == "" {
		return NativeManifest{}, fmt.Errorf("native training output has no runnable backend recommendation")
	}
	runArgs := []string{"nego", "run", "--backend", backend, opts.BaseModel, "--adapter", result.AdapterPath, "Hello"}
	chatArgs := []string{"nego", "chat", "--backend", backend, opts.BaseModel, "--adapter", result.AdapterPath, "Hello"}
	return NativeManifest{
		Version:            1,
		Type:               "nego-native-adapter",
		BaseModel:          opts.BaseModel,
		AdapterPath:        result.AdapterPath,
		Method:             result.Method,
		DatasetFormat:      result.DatasetFormat,
		TrainFile:          opts.TrainFile,
		EvalFile:           opts.EvalFile,
		RecommendedBackend: backend,
		RuntimeOptions:     map[string]string{"adapter_path": result.AdapterPath},
		RunArgs:            runArgs,
		ChatArgs:           chatArgs,
		VocabSize:          result.VocabSize,
		UpdatedTokens:      result.UpdatedTokens,
		TrainTokens:        result.TrainTokens,
		CreatedAt:          time.Now().UTC(),
	}, nil
}

func nativeTrainingRuntimeBackend(artifact *modelinfo.Artifact) string {
	if artifact == nil {
		return "native"
	}
	switch artifact.Format {
	case modelinfo.ArtifactFormatHFSafetensors:
		return "native-hf"
	case modelinfo.ArtifactFormatGGUF, modelinfo.ArtifactFormatMixed:
		return "native"
	default:
		return artifact.RecommendedRunBackend
	}
}

func writeNativeManifest(path string, manifest NativeManifest) error {
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}

func LoadNativeManifest(path string) (NativeManifest, error) {
	manifestPath, err := NativeManifestPath(path)
	if err != nil {
		return NativeManifest{}, err
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return NativeManifest{}, err
	}
	if len(data) > maxNativeManifestBytes {
		return NativeManifest{}, fmt.Errorf("native training manifest exceeds %d bytes", maxNativeManifestBytes)
	}
	var manifest NativeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return NativeManifest{}, err
	}
	if err := manifest.Validate(); err != nil {
		return NativeManifest{}, err
	}
	return manifest, nil
}

func NativeManifestPath(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("native training manifest path is required")
	}
	stat, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if stat.IsDir() {
		return filepath.Join(path, "manifest.json"), nil
	}
	if filepath.Base(path) != "manifest.json" {
		return "", os.ErrNotExist
	}
	return path, nil
}

func (m NativeManifest) Validate() error {
	if m.Version != 1 {
		return fmt.Errorf("unsupported native training manifest version %d", m.Version)
	}
	if m.Type != "nego-native-adapter" {
		return fmt.Errorf("unsupported native training manifest type %q", m.Type)
	}
	if strings.TrimSpace(m.BaseModel) == "" {
		return fmt.Errorf("native training manifest base_model is required")
	}
	if strings.TrimSpace(m.AdapterPath) == "" {
		return fmt.Errorf("native training manifest adapter_path is required")
	}
	if strings.TrimSpace(m.RecommendedBackend) == "" {
		return fmt.Errorf("native training manifest recommended_backend is required")
	}
	return nil
}

func writeNativeTrainingReadme(path string, manifest NativeManifest) error {
	var b strings.Builder
	fmt.Fprintln(&b, "# Nego Native Adapter")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Base model: `%s`\n", manifest.BaseModel)
	fmt.Fprintf(&b, "Adapter: `%s`\n", manifest.AdapterPath)
	fmt.Fprintf(&b, "Backend: `%s`\n", manifest.RecommendedBackend)
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "Run:")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "```bash")
	fmt.Fprintln(&b, strings.Join(shellQuoteArgs(manifest.RunArgs), " "))
	fmt.Fprintln(&b, "```")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "Chat:")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "```bash")
	fmt.Fprintln(&b, strings.Join(shellQuoteArgs(manifest.ChatArgs), " "))
	fmt.Fprintln(&b, "```")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func shellQuoteArgs(args []string) []string {
	out := make([]string, 0, len(args))
	for _, arg := range args {
		out = append(out, shellQuoteArg(arg))
	}
	return out
}

func shellQuoteArg(arg string) string {
	if arg == "" {
		return `""`
	}
	if strings.ContainsAny(arg, " \t\r\n\"") {
		return strconv.Quote(arg)
	}
	return arg
}

func nativeTrainingFormat(format modelinfo.ArtifactFormat) bool {
	switch format {
	case modelinfo.ArtifactFormatGGUF, modelinfo.ArtifactFormatHFSafetensors, modelinfo.ArtifactFormatMixed:
		return true
	default:
		return false
	}
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
	if opts.MaxContext < 0 {
		return opts, fmt.Errorf("max context must be greater than or equal to 0")
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

func nativeRowsTokenBudget(rows []datasets.Row, format string, tok nativeTrainingTokenizer, maxContext int) (*TokenBudgetSummary, error) {
	summary := &TokenBudgetSummary{
		Rows:       len(rows),
		MaxContext: maxContext,
		MinTokens:  -1,
	}
	for i, row := range rows {
		rowFormat := nativeBudgetFormat(row, format)
		detail := datasets.TokenBudgetRow{Row: i + 1, Format: rowFormat}
		text, err := nativeBudgetText(row, rowFormat)
		if err != nil {
			detail.Error = err.Error()
			summary.InvalidRows++
			summary.LongestRows = append(summary.LongestRows, detail)
			continue
		}
		ids, err := tok.encode(text)
		if err != nil {
			detail.Error = err.Error()
			summary.InvalidRows++
			summary.LongestRows = append(summary.LongestRows, detail)
			continue
		}
		detail.Tokens = len(ids)
		summary.CountedRows++
		summary.TotalTokens += detail.Tokens
		if summary.MinTokens < 0 || detail.Tokens < summary.MinTokens {
			summary.MinTokens = detail.Tokens
		}
		if detail.Tokens > summary.MaxTokens {
			summary.MaxTokens = detail.Tokens
		}
		if maxContext > 0 && detail.Tokens > maxContext {
			detail.OverLimit = true
			summary.OverLimit++
		}
		summary.LongestRows = append(summary.LongestRows, detail)
	}
	if summary.MinTokens < 0 {
		summary.MinTokens = 0
	}
	if summary.CountedRows > 0 {
		summary.AverageTokens = float64(summary.TotalTokens) / float64(summary.CountedRows)
	}
	summary.LongestRows = datasets.LongestTokenRows(summary.LongestRows, 5)
	if summary.InvalidRows > 0 || summary.OverLimit > 0 {
		return summary, fmt.Errorf("over_limit=%d invalid_rows=%d max_context=%d", summary.OverLimit, summary.InvalidRows, maxContext)
	}
	return summary, nil
}

func nativeBudgetFormat(row datasets.Row, format string) string {
	format = strings.ToLower(strings.TrimSpace(format))
	if format != "" && format != "auto" {
		return format
	}
	switch {
	case row["messages"] != nil:
		return "chat"
	case row["prompt"] != nil && (row["completion"] != nil || row["response"] != nil):
		return "completion"
	case row["instruction"] != nil && row["output"] != nil:
		return "instruction"
	default:
		return "unknown"
	}
}

func nativeBudgetText(row datasets.Row, format string) (string, error) {
	switch format {
	case "completion":
		prompt := rowString(row, "prompt")
		completion := rowString(row, "completion", "response")
		if prompt == "" || completion == "" {
			return "", fmt.Errorf("completion row requires prompt and completion or response")
		}
		return prompt + "\n" + completion, nil
	case "instruction":
		instruction := rowString(row, "instruction")
		output := rowString(row, "output")
		if instruction == "" || output == "" {
			return "", fmt.Errorf("instruction row requires instruction and output")
		}
		input := rowString(row, "input")
		if input != "" {
			return "Instruction: " + instruction + "\nInput: " + input + "\nOutput: " + output, nil
		}
		return "Instruction: " + instruction + "\nOutput: " + output, nil
	case "chat":
		text := chatBudgetText(row)
		if text == "" {
			return "", fmt.Errorf("chat row requires messages")
		}
		return text, nil
	default:
		return "", fmt.Errorf("unknown row format")
	}
}

func chatBudgetText(row datasets.Row) string {
	raw, ok := row["messages"].([]any)
	if !ok {
		return ""
	}
	var b strings.Builder
	for _, value := range raw {
		message, ok := value.(map[string]any)
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		content, _ := message["content"].(string)
		if strings.TrimSpace(role) == "" || strings.TrimSpace(content) == "" {
			continue
		}
		fmt.Fprintf(&b, "%s: %s\n", strings.ToUpper(role), content)
	}
	return b.String()
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
