package nativehf

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/modelinfo"
	"github.com/gakon/nego-ai/tokenizer"
)

const BackendName = "native-hf"

func init() {
	_ = nego.RegisterBackend(BackendName, Backend{})
}

type Backend struct{}

func (b Backend) Info() nego.BackendInfo {
	return nego.BackendInfo{
		Name:         BackendName,
		Description:  "Experimental pure-Go Hugging Face safetensors runtime loader.",
		Capabilities: []string{"load_safetensors", "inspect_tensors", "load_tokenizer", "load_adapter", "generate_experimental", "stream_generate", "stream_chat"},
		Required:     []string{"path"},
		Options: []nego.BackendOption{
			{Name: "output_lora", Description: "Nego output-head LoRA checkpoint matching this base model and vocabulary"},
			{Name: "adapter", Description: "token-bias adapter JSON produced by native training"},
			{Name: "adapter_path", Description: "alias for adapter"},
			{Name: "experimental_generation", Description: "enable or disable the experimental native-hf generation path"},
			{Name: "cache_tensors", Description: "cache float32 tensors in memory during generation; set false to reduce RAM use"},
			{Name: "max_tensor_cache_bytes", Description: "maximum decoded float32 tensor cache bytes per model instance; 0 means unlimited"},
		},
	}
}

func (b Backend) Load(_ context.Context, opts nego.ModelOptions) (nego.Model, error) {
	if strings.TrimSpace(opts.Path) == "" {
		return nil, fmt.Errorf("native-hf model path is required")
	}
	info, err := modelinfo.Inspect(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("inspect native-hf model: %w", err)
	}
	if info.Safetensors == nil {
		return nil, fmt.Errorf("native-hf requires safetensors weights")
	}
	if info.HFWeights == nil {
		return nil, fmt.Errorf("native-hf requires a Hugging Face weight manifest")
	}
	if err := validateNativeHFReadiness(info); err != nil {
		return nil, err
	}
	store, err := modelinfo.OpenSafetensors(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors store: %w", err)
	}
	tok, err := tokenizer.Load(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	adapter, err := loadAdapter(opts.Options)
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
		if uint64(outputLoRA.InputSize) != info.HFSpec.EmbeddingLength || uint64(outputLoRA.OutputSize) != info.HFSpec.VocabSize {
			return nil, fmt.Errorf("output LoRA dimensions do not match model embedding and vocabulary")
		}
	}
	return &Model{
		path:       opts.Path,
		info:       info,
		spec:       info.HFSpec,
		store:      store,
		tokenizer:  tok,
		float32:    make(map[string]cachedFloat32Tensor),
		cache:      optionBoolDefault(opts.Options, "cache_tensors", true),
		cacheLimit: maxCacheBytes,
		adapter:    adapter,
		outputLoRA: outputLoRA,
		generate:   optionBoolDefault(opts.Options, "experimental_generation", true),
	}, nil
}

type cachedFloat32Tensor struct {
	values []float32
	tensor modelinfo.SafetensorsTensor
}

type Model struct {
	path       string
	info       *modelinfo.Info
	spec       *modelinfo.HFModelSpec
	store      *modelinfo.SafetensorsStore
	tokenizer  *tokenizer.Tokenizer
	float32    map[string]cachedFloat32Tensor
	tensorMu   sync.Mutex
	cache      bool
	cacheBytes uint64
	cacheLimit uint64
	adapter    *adapters.TokenBiasAdapter
	outputLoRA *adapters.LinearLoRA
	generate   bool
}

func (m *Model) Generate(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	if !m.generate {
		return nil, m.inferenceError()
	}
	return m.generateText(ctx, req)
}

func (m *Model) StreamGenerate(ctx context.Context, req nego.GenerateRequest) (nego.Stream, error) {
	if !m.generate {
		return nil, m.inferenceError()
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	stream := newNativeHFStream(cancel)
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
	out, err := m.Generate(ctx, nego.GenerateRequest{
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
	m.store = nil
	m.float32 = nil
	return nil
}

func (m *Model) Info() *modelinfo.Info {
	return m.info
}

func (m *Model) Spec() *modelinfo.HFModelSpec {
	return m.spec
}

func (m *Model) Store() *modelinfo.SafetensorsStore {
	return m.store
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tokenizer
}

func (m *Model) Adapter() *adapters.TokenBiasAdapter {
	return m.adapter
}

func (m *Model) RuntimeStats() nego.RuntimeStats {
	m.tensorMu.Lock()
	defer m.tensorMu.Unlock()
	return nego.RuntimeStats{
		Backend:                BackendName,
		Path:                   m.path,
		Device:                 "cpu",
		TensorCacheEnabled:     m.cache,
		CachedTensors:          len(m.float32),
		CachedTensorBytes:      m.cacheBytes,
		MaxTensorCacheBytes:    m.cacheLimit,
		AdapterLoaded:          m.adapter != nil || m.outputLoRA != nil,
		ExperimentalGeneration: m.generate,
	}
}

func (m *Model) TensorReader(name string) (*modelinfo.SafetensorsTensorReader, modelinfo.SafetensorsTensor, error) {
	if m.store == nil {
		return nil, modelinfo.SafetensorsTensor{}, fmt.Errorf("native-hf safetensors store is not loaded")
	}
	return m.store.TensorReader(name)
}

func (m *Model) ReadTensor(name string) ([]byte, modelinfo.SafetensorsTensor, error) {
	if m.store == nil {
		return nil, modelinfo.SafetensorsTensor{}, fmt.Errorf("native-hf safetensors store is not loaded")
	}
	return m.store.ReadTensor(name)
}

func (m *Model) LoadTensorFloat32(name string) ([]float32, modelinfo.SafetensorsTensor, error) {
	values, tensor, err := m.loadTensorFloat32Shared(name)
	if err != nil {
		return nil, modelinfo.SafetensorsTensor{}, err
	}
	return cloneFloat32(values), tensor, nil
}

func (m *Model) loadTensorFloat32Shared(name string) ([]float32, modelinfo.SafetensorsTensor, error) {
	m.tensorMu.Lock()
	defer m.tensorMu.Unlock()
	if m.store == nil {
		return nil, modelinfo.SafetensorsTensor{}, fmt.Errorf("native-hf safetensors store is not loaded")
	}
	if cached, ok := m.float32[name]; ok {
		return cached.values, cached.tensor, nil
	}
	values, tensor, err := m.store.LoadTensorFloat32(name)
	if err != nil {
		return nil, modelinfo.SafetensorsTensor{}, err
	}
	if !m.cache {
		return values, tensor, nil
	}
	bytes, err := float32TensorBytes(values)
	if err != nil {
		return nil, modelinfo.SafetensorsTensor{}, err
	}
	if m.cacheLimit > 0 {
		if m.cacheBytes > m.cacheLimit || bytes > m.cacheLimit-m.cacheBytes {
			return nil, modelinfo.SafetensorsTensor{}, fmt.Errorf("native-hf decoded tensor cache would exceed max_tensor_cache_bytes: current=%d tensor=%d limit=%d", m.cacheBytes, bytes, m.cacheLimit)
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

func cloneFloat32(values []float32) []float32 {
	if values == nil {
		return nil
	}
	out := make([]float32, len(values))
	copy(out, values)
	return out
}

func (m *Model) inferenceError() error {
	ready := false
	missing := 0
	if m.info != nil && m.info.HFWeights != nil {
		ready = m.info.HFWeights.Ready
		missing = len(m.info.HFWeights.Missing)
	}
	return fmt.Errorf("native-hf experimental generation is disabled for %q; loaded safetensors, tokenizer, and manifest (ready=%v missing_tensors=%d). Omit experimental_generation=false to run the experimental native-hf path", m.path, ready, missing)
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
		return nil, fmt.Errorf("load native-hf adapter: %w", err)
	}
	return adapter, nil
}

func optionBool(options map[string]string, key string) bool {
	switch strings.ToLower(strings.TrimSpace(options[key])) {
	case "1", "t", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}

func optionBoolDefault(options map[string]string, key string, fallback bool) bool {
	if len(options) == 0 {
		return fallback
	}
	if _, ok := options[key]; !ok {
		return fallback
	}
	switch strings.ToLower(strings.TrimSpace(options[key])) {
	case "1", "t", "true", "yes", "y", "on":
		return true
	case "0", "f", "false", "no", "n", "off":
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

func (m *Model) renderChatPrompt(messages []nego.Message) (string, error) {
	if m.path != "" && hasChatTemplateFile(m.info) {
		prompt, err := chattemplate.Render(m.path, messages, chattemplate.Options{AddGenerationPrompt: true})
		if err != nil {
			return "", err
		}
		return prompt, nil
	}
	var b strings.Builder
	for _, message := range messages {
		fmt.Fprintf(&b, "%s: %s\n", strings.ToUpper(string(message.Role)), message.Content)
	}
	b.WriteString("ASSISTANT: ")
	return b.String(), nil
}

func hasChatTemplateFile(info *modelinfo.Info) bool {
	if info == nil {
		return false
	}
	for _, file := range info.Files {
		if file.Kind == "chat_template" || file.Kind == "tokenizer_config" {
			return true
		}
	}
	return false
}

func validateNativeHFReadiness(info *modelinfo.Info) error {
	if info == nil {
		return fmt.Errorf("native-hf model info is not loaded")
	}
	if info.HFSpec == nil {
		return fmt.Errorf("native-hf requires a Hugging Face config.json model spec")
	}
	if !info.HFSpec.Ready {
		reason := info.HFSpec.ValidationError
		if reason == "" {
			reason = "Hugging Face model spec is incomplete"
		}
		return fmt.Errorf("native-hf model spec is not ready: %s", reason)
	}
	if info.HFWeights == nil {
		return fmt.Errorf("native-hf requires a Hugging Face weight manifest")
	}
	if !info.HFWeights.Ready {
		return fmt.Errorf("native-hf weight manifest is not ready: missing_tensors=%d%s", len(info.HFWeights.Missing), formatListExamples(info.HFWeights.Missing, 4))
	}
	if info.HFShapes != nil && !info.HFShapes.Ready {
		return fmt.Errorf("native-hf tensor shapes are not ready: mismatches=%d%s", len(info.HFShapes.MissingShape), formatListExamples(info.HFShapes.MissingShape, 3))
	}
	return nil
}

func formatListExamples(values []string, limit int) string {
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
