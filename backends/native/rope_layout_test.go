package native

import (
	"math"
	"reflect"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestGGUFRoPEArchitectureLayout(t *testing.T) {
	// llama.cpp llama_model_rope_type selects NORM for Llama and NEOX for
	// Qwen2/Qwen3: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp
	for _, tc := range []struct {
		arch string
		want []float32
	}{
		{"llama", []float32{-1.1426396, 1.9220756, 2.9598508, 4.0297995}},
		{"qwen2", []float32{-1.9841106, 1.9599007, 2.462378, 4.0197997}},
		{"qwen3", []float32{-1.9841106, 1.9599007, 2.462378, 4.0197997}},
	} {
		t.Run(tc.arch, func(t *testing.T) {
			spec := ModelSpec{Architecture: tc.arch, RopeTheta: 10000}
			x := []float32{1, 2, 3, 4}
			got, err := spec.applyRoPE(x, 1)
			if err != nil {
				t.Fatal(err)
			}
			assertFloat32Slice(t, got, tc.want)
			if !reflect.DeepEqual(x, []float32{1, 2, 3, 4}) {
				t.Fatal("RoPE mutated its input")
			}
			got, err = spec.applyRoPE(x, 0)
			if err != nil || !reflect.DeepEqual(got, x) {
				t.Fatalf("position zero: got %v, err %v", got, err)
			}
		})
	}
}

func TestGGUFCachedAttentionRoPELayout(t *testing.T) {
	for _, tc := range []struct {
		arch string
		norm bool
		prob float32
	}{
		{"llama", false, 0.37754067},
		{"qwen2", false, 0.2848081},
		{"qwen3", true, 0.024531928},
	} {
		t.Run(tc.arch, func(t *testing.T) {
			identity := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
			tensor := modelinfo.GGUFTensor{Shape: []uint64{4, 4}}
			weights := AttentionWeights{
				QValues: identity, KValues: identity, VValues: identity, OutValues: identity,
				QTensor: tensor, KTensor: tensor, VTensor: tensor, OutTensor: tensor,
			}
			if tc.norm {
				weights.QNorm, weights.KNorm = []float32{1, 1, 1, 1}, []float32{1, 1, 1, 1}
			}
			spec := ModelSpec{Architecture: tc.arch, EmbeddingLength: 4, AttentionHeadCount: 1, KVHeadCount: 1, RopeTheta: 10000}
			cache, err := NewKVCache(1, 4)
			if err != nil {
				t.Fatal(err)
			}
			first := []float32{1, 0, 0, 0}
			got, err := multiHeadAttentionWithCacheFloat32(first, weights, spec, 0, 0, cache)
			if err != nil {
				t.Fatal(err)
			}
			assertFloat32Slice(t, got, first)
			got, err = multiHeadAttentionWithCacheFloat32([]float32{0, 0, 1, 0}, weights, spec, 0, 1, cache)
			if err != nil {
				t.Fatal(err)
			}
			// Qwen scores: [-sin(1)/2, 1/2], multiplied by four with Q/K
			// RMSNorm. Llama adjacent pairs instead give scores [0, 1/2].
			assertFloat32Slice(t, got, []float32{tc.prob, 0, 1 - tc.prob, 0})
			keys, values, err := cache.Layer(0)
			if err != nil || len(keys) != 2 || len(values) != 2 {
				t.Fatalf("cache: keys=%v values=%v err=%v", keys, values, err)
			}
			assertFloat32Slice(t, values[1], []float32{0, 0, 1, 0})
		})
	}
}

func TestGGUFRoPELayoutsRejectInvalidInputs(t *testing.T) {
	for _, arch := range []string{"llama", "qwen2", "qwen3"} {
		t.Run(arch, func(t *testing.T) {
			for _, tc := range []struct {
				x        []float32
				position int
				theta    float64
			}{
				{nil, 0, 10000},
				{[]float32{1}, 0, 10000},
				{[]float32{1, 2}, -1, 10000},
				{[]float32{1, 2}, 0, 0},
				{[]float32{1, 2}, 0, math.NaN()},
				{[]float32{1, 2}, 0, math.Inf(1)},
				{[]float32{float32(math.NaN()), 2, 3, 4}, 1, 10000},
				{[]float32{1, 2, 3, float32(math.Inf(1))}, 1, 10000},
			} {
				spec := ModelSpec{Architecture: arch, RopeTheta: tc.theta}
				if _, err := spec.applyRoPE(tc.x, tc.position); err == nil {
					t.Fatalf("expected invalid input error for %+v", tc)
				}
			}
		})
	}
}
