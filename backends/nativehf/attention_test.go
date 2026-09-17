package nativehf

import (
	"math"
	"reflect"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestHFRoPEUsesSplitHalfLayout(t *testing.T) {
	// Reference layout: Transformers v4.51.3 Qwen3/Llama rotate_half.
	// https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py#L91-L121
	x := []float32{1, 2, 3, 4}
	original := append([]float32(nil), x...)
	got, err := applyRoPEFloat32(x, 1, 10000)
	if err != nil {
		t.Fatal(err)
	}
	// Pair (1,3) at angle 1 and (2,4) at angle 0.01, preserving HF layout.
	want := []float32{-1.9841106, 1.9599007, 2.462378, 4.0197997}
	if !closeFloats(got, want) {
		t.Fatalf("RoPE = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(x, original) {
		t.Fatal("RoPE changed the input")
	}
	got, err = applyRoPEFloat32(x, 0, 10000)
	if err != nil || !reflect.DeepEqual(got, x) {
		t.Fatalf("position zero: got %v, error %v", got, err)
	}
}

func TestHFRoPERejectsInvalidInput(t *testing.T) {
	for _, tc := range []struct {
		name     string
		x        []float32
		position int
		theta    float64
	}{
		{"empty", nil, 0, 10000},
		{"odd", []float32{1, 2, 3}, 0, 10000},
		{"negative position", []float32{1, 2}, -1, 10000},
		{"zero theta", []float32{1, 2}, 0, 0},
		{"negative theta", []float32{1, 2}, 0, -1},
		{"nan theta", []float32{1, 2}, 0, math.NaN()},
		{"infinite theta", []float32{1, 2}, 0, math.Inf(1)},
		{"nan first half", []float32{float32(math.NaN()), 2, 3, 4}, 1, 10000},
		{"infinite second half", []float32{1, 2, 3, float32(math.Inf(1))}, 1, 10000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := applyRoPEFloat32(tc.x, tc.position, tc.theta); err == nil {
				t.Fatal("expected invalid RoPE input to fail")
			}
		})
	}
}

func TestHFCachedAttentionUsesSplitHalfRoPE(t *testing.T) {
	for _, tc := range []struct {
		name string
		norm bool
		prob float32
	}{
		{"llama", false, 0.2848081},
		{"qwen3", true, 0.024531928},
	} {
		t.Run(tc.name, func(t *testing.T) {
			identity := TensorValues{
				Tensor: modelinfo.SafetensorsTensor{Shape: []uint64{4, 4}},
				Values: []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
			}
			weights := AttentionWeights{Q: identity, K: identity, V: identity, Out: identity}
			if tc.norm {
				norm := &TensorValues{Values: []float32{1, 1, 1, 1}}
				weights.QNorm, weights.KNorm = norm, norm
			}
			spec := modelinfo.HFModelSpec{
				EmbeddingLength: 4, BlockCount: 1, AttentionHeadCount: 1,
				KVHeadCount: 1, HeadDim: 4, RopeTheta: 10000,
			}
			state, err := NewDecodeState(spec)
			if err != nil {
				t.Fatal(err)
			}
			first := []float32{1, 0, 0, 0}
			got, err := multiHeadAttentionFloat32(first, weights, spec, 0, 0, state)
			if err != nil || !closeFloats(got, first) {
				t.Fatalf("first attention = %v, error %v", got, err)
			}
			got, err = multiHeadAttentionFloat32([]float32{0, 0, 1, 0}, weights, spec, 0, 1, state)
			if err != nil {
				t.Fatal(err)
			}
			// Unnormalized scores: [-sin(1)/2, 1/2]. Q/K RMSNorm multiplies
			// both vectors by 2, giving [-2*sin(1), 2] before softmax.
			want := []float32{tc.prob, 0, 1 - tc.prob, 0}
			if !closeFloats(got, want) || state.CachedTokens(0, 0) != 2 {
				t.Fatalf("cached attention = %v, want %v", got, want)
			}
		})
	}
}
