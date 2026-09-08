package modelinfo

import (
	"os"
)

type MemoryOptions struct {
	ContextLength uint64
	KVBytes       uint64
}

type MemoryEstimate struct {
	Path               string         `json:"path"`
	Format             ArtifactFormat `json:"format"`
	ContextLength      uint64         `json:"context_length,omitempty"`
	EmbeddingLength    uint64         `json:"embedding_length,omitempty"`
	BlockCount         uint64         `json:"block_count,omitempty"`
	AttentionHeadCount uint64         `json:"attention_head_count,omitempty"`
	KVHeadCount        uint64         `json:"kv_head_count,omitempty"`
	HeadDim            uint64         `json:"head_dim,omitempty"`
	WeightBytes        uint64         `json:"weight_bytes,omitempty"`
	KVCacheBytes       uint64         `json:"kv_cache_bytes,omitempty"`
	RuntimeBytes       uint64         `json:"runtime_bytes,omitempty"`
	TotalBytes         uint64         `json:"total_bytes,omitempty"`
	Notes              []string       `json:"notes,omitempty"`
}

func EstimateMemory(path string, opts MemoryOptions) (*MemoryEstimate, error) {
	info, err := Inspect(path)
	if err != nil {
		return nil, err
	}
	estimate := &MemoryEstimate{
		Path:   info.Path,
		Format: artifactFormat(info),
	}
	if opts.KVBytes == 0 {
		opts.KVBytes = 4
	}
	switch {
	case info.HFSpec != nil:
		applyHFMemory(estimate, info, opts)
	case info.GGUF != nil:
		applyGGUFMemory(estimate, info.GGUF, opts)
	default:
		estimate.Notes = append(estimate.Notes, "no local model weights with runtime dimensions detected")
	}
	total, ok := checkedSum(estimate.WeightBytes, estimate.KVCacheBytes, estimate.RuntimeBytes)
	if !ok {
		estimate.Notes = append(estimate.Notes, "total memory estimate overflowed")
	} else {
		estimate.TotalBytes = total
	}
	return estimate, nil
}

func applyHFMemory(estimate *MemoryEstimate, info *Info, opts MemoryOptions) {
	spec := info.HFSpec
	estimate.ContextLength = firstPositive(opts.ContextLength, spec.ContextLength)
	estimate.EmbeddingLength = spec.EmbeddingLength
	estimate.BlockCount = spec.BlockCount
	estimate.AttentionHeadCount = spec.AttentionHeadCount
	estimate.KVHeadCount = spec.KVHeadCount
	estimate.HeadDim = spec.HeadDim
	if info.Safetensors != nil {
		estimate.WeightBytes = info.Safetensors.TotalSize
	}
	addKVEstimate(estimate, opts.KVBytes)
	if !spec.Ready {
		estimate.Notes = append(estimate.Notes, "HF model spec is incomplete; KV cache estimate may be unavailable")
	}
}

func applyGGUFMemory(estimate *MemoryEstimate, info *GGUFInfo, opts MemoryOptions) {
	estimate.ContextLength = firstPositive(opts.ContextLength, info.ContextLength)
	estimate.EmbeddingLength = info.EmbeddingLength
	estimate.BlockCount = info.BlockCount
	estimate.AttentionHeadCount = configUint(info.Metadata, info.Architecture+".attention.head_count")
	estimate.KVHeadCount = configUint(info.Metadata, info.Architecture+".attention.head_count_kv")
	if estimate.KVHeadCount == 0 {
		estimate.KVHeadCount = estimate.AttentionHeadCount
	}
	if estimate.AttentionHeadCount > 0 && estimate.EmbeddingLength%estimate.AttentionHeadCount == 0 {
		estimate.HeadDim = estimate.EmbeddingLength / estimate.AttentionHeadCount
	}
	if stat, err := os.Stat(info.Path); err == nil && stat.Size() > 0 {
		estimate.WeightBytes = uint64(stat.Size())
	}
	addKVEstimate(estimate, opts.KVBytes)
	if estimate.AttentionHeadCount == 0 || estimate.HeadDim == 0 {
		estimate.Notes = append(estimate.Notes, "GGUF attention metadata is incomplete; KV cache estimate may be unavailable")
	}
}

func addKVEstimate(estimate *MemoryEstimate, kvBytes uint64) {
	if estimate.ContextLength == 0 || estimate.BlockCount == 0 || estimate.KVHeadCount == 0 || estimate.HeadDim == 0 {
		return
	}
	values, ok := checkedProduct(estimate.ContextLength, estimate.BlockCount, estimate.KVHeadCount, estimate.HeadDim, 2, kvBytes)
	if !ok {
		estimate.Notes = append(estimate.Notes, "KV cache estimate overflowed")
		return
	}
	estimate.KVCacheBytes = values
}

func checkedProduct(values ...uint64) (uint64, bool) {
	total := uint64(1)
	for _, value := range values {
		if value == 0 {
			return 0, true
		}
		if total > ^uint64(0)/value {
			return 0, false
		}
		total *= value
	}
	return total, true
}

func checkedSum(values ...uint64) (uint64, bool) {
	var total uint64
	for _, value := range values {
		if total > ^uint64(0)-value {
			return 0, false
		}
		total += value
	}
	return total, true
}

func firstPositive(values ...uint64) uint64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}
