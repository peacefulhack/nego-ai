package modelinfo

import "fmt"

type HFWeightManifest struct {
	Architecture   string               `json:"architecture,omitempty"`
	BlockCount     int                  `json:"block_count,omitempty"`
	TokenEmbedding string               `json:"token_embedding,omitempty"`
	OutputNorm     string               `json:"output_norm,omitempty"`
	Output         string               `json:"output,omitempty"`
	TiedOutput     bool                 `json:"tied_output,omitempty"`
	Blocks         []HFBlockTensorNames `json:"blocks,omitempty"`
	Missing        []string             `json:"missing,omitempty"`
	Ready          bool                 `json:"ready"`
}

type HFBlockTensorNames struct {
	InputNorm    string `json:"input_norm,omitempty"`
	AttentionQ   string `json:"attention_q,omitempty"`
	AttentionK   string `json:"attention_k,omitempty"`
	AttentionV   string `json:"attention_v,omitempty"`
	AttentionOut string `json:"attention_out,omitempty"`
	PostNorm     string `json:"post_norm,omitempty"`
	FFNGate      string `json:"ffn_gate,omitempty"`
	FFNUp        string `json:"ffn_up,omitempty"`
	FFNDown      string `json:"ffn_down,omitempty"`
}

func BuildHFWeightManifest(path string) (*HFWeightManifest, error) {
	info, err := Inspect(path)
	if err != nil {
		return nil, err
	}
	return HFWeightManifestFromInfo(info)
}

func HFWeightManifestFromInfo(info *Info) (*HFWeightManifest, error) {
	if info == nil {
		return nil, fmt.Errorf("model info is nil")
	}
	if info.Safetensors == nil {
		return nil, fmt.Errorf("model has no safetensors metadata")
	}
	blockCount := int(configUint(info.Config, "num_hidden_layers", "n_layers", "num_layers"))
	manifest := &HFWeightManifest{
		Architecture: firstArchitecture(info),
		BlockCount:   blockCount,
		Blocks:       make([]HFBlockTensorNames, blockCount),
	}
	tensors := safetensorsTensorSet(info.Safetensors)
	manifest.TokenEmbedding = requireHFTensor(tensors, &manifest.Missing, "model.embed_tokens.weight", "transformer.wte.weight")
	manifest.OutputNorm = requireHFTensor(tensors, &manifest.Missing, "model.norm.weight", "transformer.ln_f.weight")
	manifest.Output = optionalHFTensor(tensors, "lm_head.weight")
	if manifest.Output == "" {
		manifest.TiedOutput = true
		manifest.Output = manifest.TokenEmbedding
	}
	for i := 0; i < blockCount; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		altPrefix := fmt.Sprintf("transformer.h.%d.", i)
		manifest.Blocks[i] = HFBlockTensorNames{
			InputNorm:    requireHFTensor(tensors, &manifest.Missing, prefix+"input_layernorm.weight", altPrefix+"ln_1.weight"),
			AttentionQ:   requireHFTensor(tensors, &manifest.Missing, prefix+"self_attn.q_proj.weight", altPrefix+"attn.q_proj.weight"),
			AttentionK:   requireHFTensor(tensors, &manifest.Missing, prefix+"self_attn.k_proj.weight", altPrefix+"attn.k_proj.weight"),
			AttentionV:   requireHFTensor(tensors, &manifest.Missing, prefix+"self_attn.v_proj.weight", altPrefix+"attn.v_proj.weight"),
			AttentionOut: requireHFTensor(tensors, &manifest.Missing, prefix+"self_attn.o_proj.weight", altPrefix+"attn.o_proj.weight"),
			PostNorm:     requireHFTensor(tensors, &manifest.Missing, prefix+"post_attention_layernorm.weight", altPrefix+"ln_2.weight"),
			FFNGate:      requireHFTensor(tensors, &manifest.Missing, prefix+"mlp.gate_proj.weight", altPrefix+"mlp.gate_proj.weight"),
			FFNUp:        requireHFTensor(tensors, &manifest.Missing, prefix+"mlp.up_proj.weight", altPrefix+"mlp.up_proj.weight"),
			FFNDown:      requireHFTensor(tensors, &manifest.Missing, prefix+"mlp.down_proj.weight", altPrefix+"mlp.down_proj.weight"),
		}
	}
	manifest.Ready = blockCount > 0 && len(manifest.Missing) == 0
	return manifest, nil
}

func safetensorsTensorSet(info *SafetensorsInfo) map[string]bool {
	out := make(map[string]bool, len(info.Tensors))
	for _, tensor := range info.Tensors {
		out[tensor.Name] = true
	}
	return out
}

func requireHFTensor(tensors map[string]bool, missing *[]string, names ...string) string {
	if name := optionalHFTensor(tensors, names...); name != "" {
		return name
	}
	if len(names) > 0 {
		*missing = append(*missing, names[0])
	}
	return ""
}

func optionalHFTensor(tensors map[string]bool, names ...string) string {
	for _, name := range names {
		if tensors[name] {
			return name
		}
	}
	return ""
}

func firstArchitecture(info *Info) string {
	if len(info.Architectures) > 0 {
		return info.Architectures[0]
	}
	return info.ModelType
}

func configUint(config map[string]any, keys ...string) uint64 {
	for _, key := range keys {
		switch value := config[key].(type) {
		case uint64:
			return value
		case uint32:
			return uint64(value)
		case int:
			if value >= 0 {
				return uint64(value)
			}
		case int64:
			if value >= 0 {
				return uint64(value)
			}
		case float64:
			if value >= 0 {
				return uint64(value)
			}
		}
	}
	return 0
}
