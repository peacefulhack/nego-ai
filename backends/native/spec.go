package native

import (
	"fmt"
	"math"

	"github.com/gakon/nego-ai/modelinfo"
)

type ModelSpec struct {
	Architecture       string
	ContextLength      uint64
	EmbeddingLength    uint64
	BlockCount         uint64
	FeedForwardLength  uint64
	AttentionHeadCount uint64
	KVHeadCount        uint64
	RopeTheta          float64
	RMSNormEpsilon     float32
}

type BlockTensorNames struct {
	AttentionNorm string
	AttentionQ    string
	AttentionK    string
	AttentionV    string
	AttentionOut  string
	FFNNorm       string
	FFNGate       string
	FFNUp         string
	FFNDown       string
}

type TensorNames struct {
	TokenEmbedding string
	OutputNorm     string
	Output         string
	Blocks         []BlockTensorNames
}

func buildModelSpec(info *modelinfo.GGUFInfo) (ModelSpec, TensorNames, error) {
	if info == nil {
		return ModelSpec{}, TensorNames{}, fmt.Errorf("gguf info is nil")
	}
	spec := ModelSpec{
		Architecture:       info.Architecture,
		ContextLength:      info.ContextLength,
		EmbeddingLength:    info.EmbeddingLength,
		BlockCount:         info.BlockCount,
		FeedForwardLength:  metadataUint(info.Metadata, info.Architecture+".feed_forward_length"),
		AttentionHeadCount: metadataUint(info.Metadata, info.Architecture+".attention.head_count"),
		KVHeadCount:        metadataUint(info.Metadata, info.Architecture+".attention.head_count_kv"),
		RopeTheta:          metadataFloat(info.Metadata, info.Architecture+".rope.freq_base", 10000),
		RMSNormEpsilon:     float32(metadataFloat(info.Metadata, info.Architecture+".attention.layer_norm_rms_epsilon", 1e-5)),
	}
	if spec.KVHeadCount == 0 {
		spec.KVHeadCount = spec.AttentionHeadCount
	}
	if err := spec.validate(); err != nil {
		return ModelSpec{}, TensorNames{}, err
	}
	return spec, tensorNames(spec), nil
}

func (s ModelSpec) validate() error {
	if s.Architecture == "" {
		return fmt.Errorf("native model architecture is missing")
	}
	if s.EmbeddingLength == 0 {
		return fmt.Errorf("native model embedding length is missing")
	}
	if s.BlockCount == 0 {
		return fmt.Errorf("native model block count is missing")
	}
	if s.BlockCount > uint64(int(^uint(0)>>1)) {
		return fmt.Errorf("native model block count overflows this runtime")
	}
	if s.AttentionHeadCount == 0 {
		return fmt.Errorf("native model attention head count is missing")
	}
	if s.EmbeddingLength%s.AttentionHeadCount != 0 {
		return fmt.Errorf("embedding length %d is not divisible by attention head count %d", s.EmbeddingLength, s.AttentionHeadCount)
	}
	if s.KVHeadCount == 0 || s.AttentionHeadCount%s.KVHeadCount != 0 {
		return fmt.Errorf("attention head count %d is not divisible by KV head count %d", s.AttentionHeadCount, s.KVHeadCount)
	}
	if s.FeedForwardLength == 0 {
		return fmt.Errorf("native model feed-forward length is missing")
	}
	if s.RopeTheta <= 0 || math.IsNaN(s.RopeTheta) || math.IsInf(s.RopeTheta, 0) {
		return fmt.Errorf("native model RoPE theta must be finite and positive")
	}
	if s.RMSNormEpsilon < 0 || math.IsNaN(float64(s.RMSNormEpsilon)) || math.IsInf(float64(s.RMSNormEpsilon), 0) {
		return fmt.Errorf("native model RMSNorm epsilon must be finite and non-negative")
	}
	return nil
}

func tensorNames(spec ModelSpec) TensorNames {
	out := TensorNames{
		TokenEmbedding: "token_embd.weight",
		OutputNorm:     "output_norm.weight",
		Output:         "output.weight",
		Blocks:         make([]BlockTensorNames, int(spec.BlockCount)),
	}
	for i := range out.Blocks {
		prefix := fmt.Sprintf("blk.%d.", i)
		out.Blocks[i] = BlockTensorNames{
			AttentionNorm: prefix + "attn_norm.weight",
			AttentionQ:    prefix + "attn_q.weight",
			AttentionK:    prefix + "attn_k.weight",
			AttentionV:    prefix + "attn_v.weight",
			AttentionOut:  prefix + "attn_output.weight",
			FFNNorm:       prefix + "ffn_norm.weight",
			FFNGate:       prefix + "ffn_gate.weight",
			FFNUp:         prefix + "ffn_up.weight",
			FFNDown:       prefix + "ffn_down.weight",
		}
	}
	return out
}

func metadataUint(metadata map[string]any, key string) uint64 {
	switch value := metadata[key].(type) {
	case uint8:
		return uint64(value)
	case uint16:
		return uint64(value)
	case uint32:
		return uint64(value)
	case uint64:
		return value
	case int8:
		if value >= 0 {
			return uint64(value)
		}
	case int16:
		if value >= 0 {
			return uint64(value)
		}
	case int32:
		if value >= 0 {
			return uint64(value)
		}
	case int64:
		if value >= 0 {
			return uint64(value)
		}
	}
	return 0
}

func metadataFloat(metadata map[string]any, key string, fallback float64) float64 {
	switch value := metadata[key].(type) {
	case uint8:
		return float64(value)
	case uint16:
		return float64(value)
	case uint32:
		return float64(value)
	case uint64:
		return float64(value)
	case int8:
		return float64(value)
	case int16:
		return float64(value)
	case int32:
		return float64(value)
	case int64:
		return float64(value)
	case float32:
		return float64(value)
	case float64:
		return value
	default:
		return fallback
	}
}
