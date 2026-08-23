package native

import (
	"strings"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestBuildModelSpec(t *testing.T) {
	info := &modelinfo.GGUFInfo{
		Architecture:    "llama",
		ContextLength:   128,
		EmbeddingLength: 16,
		BlockCount:      2,
		Metadata: map[string]any{
			"llama.feed_forward_length":              uint32(32),
			"llama.attention.head_count":             uint32(4),
			"llama.attention.head_count_kv":          uint32(2),
			"llama.rope.freq_base":                   float32(10000),
			"llama.attention.layer_norm_rms_epsilon": float32(1e-6),
		},
	}
	spec, names, err := buildModelSpec(info)
	if err != nil {
		t.Fatal(err)
	}
	if spec.FeedForwardLength != 32 || spec.KVHeadCount != 2 || spec.RMSNormEpsilon != 1e-6 {
		t.Fatalf("unexpected spec: %#v", spec)
	}
	if len(names.Blocks) != 2 || names.Blocks[1].FFNDown != "blk.1.ffn_down.weight" {
		t.Fatalf("unexpected tensor names: %#v", names)
	}
}

func TestBuildModelSpecDefaultsKVHeads(t *testing.T) {
	info := &modelinfo.GGUFInfo{
		Architecture:    "llama",
		EmbeddingLength: 16,
		BlockCount:      1,
		Metadata: map[string]any{
			"llama.feed_forward_length":  uint32(32),
			"llama.attention.head_count": uint32(4),
		},
	}
	spec, _, err := buildModelSpec(info)
	if err != nil {
		t.Fatal(err)
	}
	if spec.KVHeadCount != 4 || spec.RopeTheta != 10000 || spec.RMSNormEpsilon != 1e-5 {
		t.Fatalf("unexpected defaults: %#v", spec)
	}
}

func TestBuildModelSpecRejectsInvalidDimensions(t *testing.T) {
	tests := []struct {
		name string
		info *modelinfo.GGUFInfo
		want string
	}{
		{
			name: "missing architecture",
			info: &modelinfo.GGUFInfo{},
			want: "architecture",
		},
		{
			name: "bad heads",
			info: specInfo(15, 4, 2),
			want: "divisible",
		},
		{
			name: "bad kv heads",
			info: specInfo(16, 4, 3),
			want: "KV",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := buildModelSpec(tc.info)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected %q error, got %v", tc.want, err)
			}
		})
	}
}

func specInfo(embed, heads, kvHeads uint64) *modelinfo.GGUFInfo {
	return &modelinfo.GGUFInfo{
		Architecture:    "llama",
		EmbeddingLength: embed,
		BlockCount:      1,
		Metadata: map[string]any{
			"llama.feed_forward_length":     uint32(32),
			"llama.attention.head_count":    heads,
			"llama.attention.head_count_kv": kvHeads,
		},
	}
}
