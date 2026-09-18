package native

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
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
		Description:  "Experimental pure-Go GGUF runtime foundation for local model loading, tokenization, and early CPU inference.",
		Capabilities: []string{"load_gguf", "inspect_tensors", "tokenize_gguf", "generate_experimental", "stream_generate", "chat_experimental", "stream_chat", "kv_cache", "float32_tensor_cache", "legacy_quant_dequant", "k_quant_dequant", "qk_norm_attention"},
		Required:     []string{"path"},
		Options: []nego.BackendOption{
			{Name: "output_lora", Description: "Nego linear LoRA checkpoint for the output projection; must match this base model and vocabulary"},
			{Name: "template_path", Description: "directory or file path for chat template sidecars"},
			{Name: "adapter", Description: "token-bias adapter JSON produced by native training"},
			{Name: "adapter_path", Description: "alias for adapter"},
			{Name: "allow_incomplete", Description: "allow loading incomplete GGUF manifests for tensor inspection; generation may still fail"},
			{Name: "cache_tensors", Description: "cache decoded float32 tensors per model instance; defaults to true"},
			{Name: "max_tensor_read_bytes", Description: "maximum raw tensor bytes read into memory at once; defaults to 536870912"},
			{Name: "max_tensor_cache_bytes", Description: "maximum decoded float32 tensor cache bytes per model instance; 0 means unlimited"},
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
	vocab, err := modelinfo.InspectGGUFVocab(modelPath)
	if err != nil {
		return nil, fmt.Errorf("inspect native GGUF vocab: %w", err)
	}
	spec, tensorNames, err := buildModelSpec(info)
	if err != nil {
		return nil, fmt.Errorf("build native model spec: %w", err)
	}
	manifest := buildTensorManifest(info, spec, tensorNames)
	if !optionBoolDefault(opts.Options, "allow_incomplete", false) {
		if err := validateNativeGGUFLoad(modelPath, info, manifest); err != nil {
			return nil, err
		}
	}
	maxReadBytes, err := optionUintDefault(opts.Options, "max_tensor_read_bytes", defaultMaxTensorReadBytes)
	if err != nil {
		return nil, err
	}
	maxCacheBytes, err := optionUintDefault(opts.Options, "max_tensor_cache_bytes", 0)
	if err != nil {
		return nil, err
	}
	var outputLoRA *adapters.LinearLoRA
	if path := opts.Options["output_lora"]; path != "" {
		outputLoRA, err = adapters.LoadLinearLoRA(path)
		if err != nil {
			return nil, fmt.Errorf("load output LoRA: %w", err)
		}
		if uint64(outputLoRA.InputSize) != spec.EmbeddingLength || outputLoRA.OutputSize != len(vocab.Tokens) {
			return nil, fmt.Errorf("output LoRA dimensions do not match model embedding and vocabulary")
		}
	}
	tensors, err := openTensorStore(modelPath, info, maxReadBytes)
	if err != nil {
		return nil, fmt.Errorf("open native tensor store: %w", err)
	}
	adapter, err := loadAdapter(opts.Options)
	if err != nil {
		tensors.Close()
		return nil, err
	}
	return &Model{
		path:          modelPath,
		info:          info,
		vocab:         vocab,
		spec:          spec,
		names:         tensorNames,
		manifest:      manifest,
		tensors:       tensors,
		float32:       make(map[string]cachedFloat32Tensor),
		cacheTensors:  optionBoolDefault(opts.Options, "cache_tensors", true),
		maxCacheBytes: maxCacheBytes,
		adapter:       adapter,
		outputLoRA:    outputLoRA,
		promptPath:    promptPath(opts.Path, modelPath, opts.Options),
	}, nil
}

func validateNativeGGUFLoad(modelPath string, info *modelinfo.GGUFInfo, manifest TensorManifestReport) error {
	if !manifest.Ready() {
		return fmt.Errorf("native GGUF tensor manifest is not ready for %q: missing=%d%s shape_errors=%d%s; run `nego check %s` for full details or set allow_incomplete=true for tensor inspection",
			modelPath,
			len(manifest.Missing),
			formatIssueExamples(manifest.Missing, 3),
			len(manifest.MissingShape),
			formatIssueExamples(manifest.MissingShape, 2),
			modelPath,
		)
	}
	unsupported := unsupportedRequiredTensorTypes(info, manifest)
	if len(unsupported) > 0 {
		return fmt.Errorf("native GGUF required tensor types are not ready for %q: unsupported=%d%s; run `nego check %s` for full details or set allow_incomplete=true for tensor inspection",
			modelPath,
			len(unsupported),
			formatIssueExamples(unsupported, 3),
			modelPath,
		)
	}
	return nil
}

func unsupportedRequiredTensorTypes(info *modelinfo.GGUFInfo, manifest TensorManifestReport) []string {
	required := make(map[string]struct{}, len(manifest.Required))
	for _, name := range manifest.Required {
		required[name] = struct{}{}
	}
	var unsupported []string
	for _, tensor := range info.Tensors {
		if _, ok := required[tensor.Name]; !ok || tensorFloat32SupportedType(tensor.GGMLType) {
			continue
		}
		unsupported = append(unsupported, fmt.Sprintf("%s:%s", tensor.Name, tensorTypeLabel(tensor)))
	}
	return unsupported
}

func tensorTypeLabel(tensor modelinfo.GGUFTensor) string {
	if tensor.Type != "" {
		return tensor.Type
	}
	return fmt.Sprintf("ggml_type_%d", tensor.GGMLType)
}

type cachedFloat32Tensor struct {
	values []float32
	tensor modelinfo.GGUFTensor
}

type Model struct {
	path          string
	info          *modelinfo.GGUFInfo
	vocab         *modelinfo.GGUFVocab
	spec          ModelSpec
	names         TensorNames
	manifest      TensorManifestReport
	tensors       *tensorStore
	float32       map[string]cachedFloat32Tensor
	cacheTensors  bool
	cacheBytes    uint64
	maxCacheBytes uint64
	tensorMu      sync.Mutex
	adapter       *adapters.TokenBiasAdapter
	outputLoRA    *adapters.LinearLoRA
	promptPath    string
}

func (m *Model) Generate(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	return m.generateText(ctx, req)
}

func (m *Model) StreamGenerate(ctx context.Context, req nego.GenerateRequest) (nego.Stream, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	stream := newNativeStream(cancel)
	go stream.run(ctx, func(emit func(string) error) error {
		_, err := m.generateTextWithEmitter(ctx, req, emit)
		return err
	})
	return stream, nil
}

func (m *Model) Chat(ctx context.Context, req nego.ChatRequest) (*nego.ChatResponse, error) {
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		return nil, err
	}
	out, err := m.generateText(ctx, nego.GenerateRequest{
		Prompt:        prompt,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		RepeatPenalty: req.RepeatPenalty,
		Stop:          req.Stop,
		Seed:          req.Seed,
	})
	if err != nil {
		return nil, err
	}
	return &nego.ChatResponse{Message: nego.Message{Role: nego.RoleAssistant, Content: out.Text}}, nil
}

func (m *Model) StreamChat(ctx context.Context, req nego.ChatRequest) (nego.Stream, error) {
	prompt, err := m.renderChatPrompt(req.Messages)
	if err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return m.StreamGenerate(ctx, nego.GenerateRequest{
		Prompt:        prompt,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		RepeatPenalty: req.RepeatPenalty,
		Stop:          req.Stop,
		Seed:          req.Seed,
	})
}

func (m *Model) Close() error {
	m.tensorMu.Lock()
	defer m.tensorMu.Unlock()
	if m.tensors == nil {
		return nil
	}
	err := m.tensors.Close()
	m.tensors = nil
	m.float32 = nil
	return err
}

func (m *Model) Info() *modelinfo.GGUFInfo {
	return m.info
}

func (m *Model) Vocab() *modelinfo.GGUFVocab {
	return m.vocab
}

func (m *Model) Spec() ModelSpec {
	return m.spec
}

func (m *Model) TensorNames() TensorNames {
	return m.names
}

func (m *Model) TensorManifest() TensorManifestReport {
	return m.manifest
}

func (m *Model) Adapter() *adapters.TokenBiasAdapter {
	return m.adapter
}

func (m *Model) RuntimeStats() nego.RuntimeStats {
	m.tensorMu.Lock()
	defer m.tensorMu.Unlock()
	return nego.RuntimeStats{
		Backend:             BackendName,
		Path:                m.path,
		Device:              "cpu",
		TensorCacheEnabled:  m.cacheTensors,
		CachedTensors:       len(m.float32),
		CachedTensorBytes:   m.cacheBytes,
		MaxTensorCacheBytes: m.maxCacheBytes,
		AdapterLoaded:       m.adapter != nil || m.outputLoRA != nil,
	}
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

func (m *Model) LoadTensorFloat32(name string) ([]float32, modelinfo.GGUFTensor, error) {
	values, tensor, err := m.loadTensorFloat32Shared(name)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	return cloneFloat32(values), tensor, nil
}

func (m *Model) loadTensorFloat32Shared(name string) ([]float32, modelinfo.GGUFTensor, error) {
	m.tensorMu.Lock()
	defer m.tensorMu.Unlock()
	if m.tensors == nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("native tensor store is closed")
	}
	if m.cacheTensors {
		if cached, ok := m.float32[name]; ok {
			return cached.values, cached.tensor, nil
		}
	}
	data, tensor, err := m.tensors.ReadTensor(name)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	values, err := tensorFloat32(tensor, data)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	if !m.cacheTensors {
		return values, tensor, nil
	}
	bytes, err := float32TensorBytes(values)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	if m.maxCacheBytes > 0 {
		if m.cacheBytes > m.maxCacheBytes || bytes > m.maxCacheBytes-m.cacheBytes {
			return nil, modelinfo.GGUFTensor{}, fmt.Errorf("native decoded tensor cache would exceed max_tensor_cache_bytes: current=%d tensor=%d limit=%d", m.cacheBytes, bytes, m.maxCacheBytes)
		}
	}
	if m.float32 == nil {
		m.float32 = make(map[string]cachedFloat32Tensor)
	}
	m.float32[name] = cachedFloat32Tensor{values: values, tensor: tensor}
	m.cacheBytes += bytes
	return values, tensor, nil
}

func float32TensorBytes(values []float32) (uint64, error) {
	if uint64(len(values)) > ^uint64(0)/4 {
		return 0, fmt.Errorf("decoded float32 tensor byte size overflows")
	}
	return uint64(len(values)) * 4, nil
}

func (m *Model) DecodeTokenIDs(ids []int) (string, error) {
	if m.vocab == nil {
		return "", fmt.Errorf("native GGUF vocab is not loaded")
	}
	return m.vocab.Decode(ids, modelinfo.DecodeOptions{SkipSpecial: true})
}

func (m *Model) EncodeText(text string) ([]int, error) {
	if m.vocab == nil {
		return nil, fmt.Errorf("native GGUF vocab is not loaded")
	}
	return m.vocab.Encode(text, m.vocab.DefaultEncodeOptions())
}

func (m *Model) renderChatPrompt(messages []nego.Message) (string, error) {
	if m.promptPath == "" {
		return fallbackChatPrompt(messages), nil
	}
	return chattemplate.Render(m.promptPath, messages, chattemplate.Options{AddGenerationPrompt: true})
}

func (m *Model) inferenceError() error {
	if !m.manifest.Ready() {
		return fmt.Errorf("native GGUF tensor manifest is not ready for %q: missing=%d%s shape_errors=%d%s; run `nego check %s` for full details",
			m.path,
			len(m.manifest.Missing),
			formatIssueExamples(m.manifest.Missing, 3),
			len(m.manifest.MissingShape),
			formatIssueExamples(m.manifest.MissingShape, 2),
			m.path,
		)
	}
	return fmt.Errorf("native GGUF inference could not run for architecture %q with %d tensors; run `nego check %s` for compatibility details", m.info.Architecture, len(m.info.Tensors), m.path)
}

func formatIssueExamples(values []string, limit int) string {
	if len(values) == 0 {
		return ""
	}
	if limit <= 0 || limit > len(values) {
		limit = len(values)
	}
	suffix := ""
	if remaining := len(values) - limit; remaining > 0 {
		suffix = fmt.Sprintf(", ... %d more", remaining)
	}
	return " [" + strings.Join(values[:limit], ", ") + suffix + "]"
}

func (m *Model) applyAdapter(logits []float32) []float32 {
	if m.adapter == nil {
		return logits
	}
	return m.adapter.Apply(logits)
}

func loadAdapter(options map[string]string) (*adapters.TokenBiasAdapter, error) {
	if len(options) == 0 {
		return nil, nil
	}
	path := options["adapter"]
	if path == "" {
		path = options["adapter_path"]
	}
	if path == "" {
		return nil, nil
	}
	adapter, err := adapters.Load(path)
	if err != nil {
		return nil, fmt.Errorf("load native adapter: %w", err)
	}
	return adapter, nil
}

func optionBoolDefault(options map[string]string, key string, fallback bool) bool {
	if len(options) == 0 {
		return fallback
	}
	value, ok := options[key]
	if !ok {
		return fallback
	}
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func optionUintDefault(options map[string]string, key string, fallback uint64) (uint64, error) {
	if len(options) == 0 {
		return fallback, nil
	}
	value := strings.TrimSpace(options[key])
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("option %s must be an unsigned integer", key)
	}
	return parsed, nil
}

func promptPath(inputPath, modelPath string, options map[string]string) string {
	if value := options["template_path"]; value != "" {
		return value
	}
	if inputPath != "" {
		if info, err := os.Stat(inputPath); err == nil && !info.IsDir() {
			return filepath.Dir(inputPath)
		}
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
