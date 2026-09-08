package modelinfo

import (
	"fmt"
	"math"
)

type HFModelSpec struct {
	ModelType          string   `json:"model_type,omitempty"`
	Architecture       string   `json:"architecture,omitempty"`
	VocabSize          uint64   `json:"vocab_size,omitempty"`
	ContextLength      uint64   `json:"context_length,omitempty"`
	EmbeddingLength    uint64   `json:"embedding_length,omitempty"`
	BlockCount         uint64   `json:"block_count,omitempty"`
	FeedForwardLength  uint64   `json:"feed_forward_length,omitempty"`
	AttentionHeadCount uint64   `json:"attention_head_count,omitempty"`
	KVHeadCount        uint64   `json:"kv_head_count,omitempty"`
	HeadDim            uint64   `json:"head_dim,omitempty"`
	RopeTheta          float64  `json:"rope_theta,omitempty"`
	RMSNormEpsilon     float32  `json:"rms_norm_epsilon,omitempty"`
	TieWordEmbeddings  bool     `json:"tie_word_embeddings"`
	Missing            []string `json:"missing,omitempty"`
	ValidationError    string   `json:"validation_error,omitempty"`
	Ready              bool     `json:"ready"`
}

func BuildHFModelSpec(path string) (*HFModelSpec, error) {
	info, err := Inspect(path)
	if err != nil {
		return nil, err
	}
	return HFModelSpecFromInfo(info)
}

func HFModelSpecFromInfo(info *Info) (*HFModelSpec, error) {
	if info == nil {
		return nil, fmt.Errorf("model info is nil")
	}
	if info.Config == nil {
		return nil, fmt.Errorf("model config is missing")
	}
	spec := &HFModelSpec{
		ModelType:         info.ModelType,
		Architecture:      firstArchitecture(info),
		RopeTheta:         configFloat(info.Config, 10000, "rope_theta", "rotary_emb_base"),
		RMSNormEpsilon:    float32(configFloat(info.Config, 1e-5, "rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps")),
		TieWordEmbeddings: configBool(info.Config, false, "tie_word_embeddings"),
	}
	spec.VocabSize = requiredConfigUint(info.Config, &spec.Missing, "vocab_size")
	spec.ContextLength = requiredConfigUint(info.Config, &spec.Missing, "max_position_embeddings", "n_positions", "seq_length")
	spec.EmbeddingLength = requiredConfigUint(info.Config, &spec.Missing, "hidden_size", "n_embd", "d_model")
	spec.BlockCount = requiredConfigUint(info.Config, &spec.Missing, "num_hidden_layers", "n_layers", "num_layers")
	spec.FeedForwardLength = requiredConfigUint(info.Config, &spec.Missing, "intermediate_size", "ffn_dim")
	spec.AttentionHeadCount = requiredConfigUint(info.Config, &spec.Missing, "num_attention_heads", "n_head")
	spec.KVHeadCount = configUint(info.Config, "num_key_value_heads", "n_head_kv")
	if spec.KVHeadCount == 0 {
		spec.KVHeadCount = spec.AttentionHeadCount
	}
	spec.HeadDim = configUint(info.Config, "head_dim", "attention_head_dim")
	if spec.HeadDim == 0 && spec.EmbeddingLength > 0 && spec.AttentionHeadCount > 0 && spec.EmbeddingLength%spec.AttentionHeadCount == 0 {
		spec.HeadDim = spec.EmbeddingLength / spec.AttentionHeadCount
	}
	if err := spec.Validate(); err != nil {
		spec.ValidationError = err.Error()
	}
	spec.Ready = len(spec.Missing) == 0 && spec.ValidationError == ""
	return spec, nil
}

func (s HFModelSpec) Validate() error {
	if s.ModelType == "" && s.Architecture == "" {
		return fmt.Errorf("HF model architecture metadata is missing")
	}
	if s.VocabSize == 0 {
		return fmt.Errorf("HF model vocab size is missing")
	}
	if s.ContextLength == 0 {
		return fmt.Errorf("HF model context length is missing")
	}
	if s.EmbeddingLength == 0 {
		return fmt.Errorf("HF model embedding length is missing")
	}
	if s.BlockCount == 0 {
		return fmt.Errorf("HF model block count is missing")
	}
	if s.BlockCount > uint64(int(^uint(0)>>1)) {
		return fmt.Errorf("HF model block count overflows this runtime")
	}
	if s.FeedForwardLength == 0 {
		return fmt.Errorf("HF model feed-forward length is missing")
	}
	if s.AttentionHeadCount == 0 {
		return fmt.Errorf("HF model attention head count is missing")
	}
	if s.KVHeadCount == 0 || s.AttentionHeadCount%s.KVHeadCount != 0 {
		return fmt.Errorf("HF attention head count %d is not divisible by KV head count %d", s.AttentionHeadCount, s.KVHeadCount)
	}
	if s.HeadDim == 0 {
		return fmt.Errorf("HF model attention head dimension is missing")
	}
	if s.RopeTheta <= 0 || math.IsNaN(s.RopeTheta) || math.IsInf(s.RopeTheta, 0) {
		return fmt.Errorf("HF model RoPE theta must be finite and positive")
	}
	if s.RMSNormEpsilon < 0 || math.IsNaN(float64(s.RMSNormEpsilon)) || math.IsInf(float64(s.RMSNormEpsilon), 0) {
		return fmt.Errorf("HF model RMSNorm epsilon must be finite and non-negative")
	}
	return nil
}

func requiredConfigUint(config map[string]any, missing *[]string, keys ...string) uint64 {
	value := configUint(config, keys...)
	if value == 0 && len(keys) > 0 {
		*missing = append(*missing, keys[0])
	}
	return value
}

func configFloat(config map[string]any, fallback float64, keys ...string) float64 {
	for _, key := range keys {
		switch value := config[key].(type) {
		case float64:
			return value
		case float32:
			return float64(value)
		case int:
			return float64(value)
		case int64:
			return float64(value)
		case uint64:
			return float64(value)
		case uint32:
			return float64(value)
		}
	}
	return fallback
}

func configBool(config map[string]any, fallback bool, keys ...string) bool {
	for _, key := range keys {
		if value, ok := config[key].(bool); ok {
			return value
		}
	}
	return fallback
}
