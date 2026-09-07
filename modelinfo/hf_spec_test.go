package modelinfo

import "testing"

func TestHFModelSpecFromQwenConfig(t *testing.T) {
	info := &Info{
		ModelType:     "qwen3",
		Architectures: []string{"Qwen3ForCausalLM"},
		Config: map[string]any{
			"model_type":              "qwen3",
			"architectures":           []any{"Qwen3ForCausalLM"},
			"vocab_size":              float64(151936),
			"max_position_embeddings": float64(40960),
			"hidden_size":             float64(1024),
			"num_hidden_layers":       float64(28),
			"intermediate_size":       float64(3072),
			"num_attention_heads":     float64(16),
			"num_key_value_heads":     float64(8),
			"head_dim":                float64(128),
			"rope_theta":              float64(1000000),
			"rms_norm_eps":            float64(1e-6),
			"tie_word_embeddings":     true,
		},
	}
	spec, err := HFModelSpecFromInfo(info)
	if err != nil {
		t.Fatal(err)
	}
	if !spec.Ready || spec.ValidationError != "" || len(spec.Missing) != 0 {
		t.Fatalf("unexpected spec readiness: %#v", spec)
	}
	if spec.HeadDim != 128 || spec.KVHeadCount != 8 || !spec.TieWordEmbeddings {
		t.Fatalf("unexpected qwen spec: %#v", spec)
	}
}

func TestHFModelSpecReportsMissingConfig(t *testing.T) {
	spec, err := HFModelSpecFromInfo(&Info{ModelType: "llama", Config: map[string]any{
		"hidden_size": float64(128),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if spec.Ready || len(spec.Missing) == 0 || spec.ValidationError == "" {
		t.Fatalf("expected incomplete spec: %#v", spec)
	}
}
