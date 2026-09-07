package nativehf

import (
	"math"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestHFEmbeddingLookupUsesRowsAsTokens(t *testing.T) {
	tensor := modelinfo.SafetensorsTensor{Name: "model.embed_tokens.weight", Shape: []uint64{2, 3}}
	got, err := embeddingLookupFloat32([]float32{1, 2, 3, 4, 5, 6}, tensor, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{4, 5, 6}
	if !closeFloats(got, want) {
		t.Fatalf("embedding = %v, want %v", got, want)
	}
}

func TestHFLinearUsesRowsAsOutputs(t *testing.T) {
	tensor := modelinfo.SafetensorsTensor{Name: "proj.weight", Shape: []uint64{2, 2}}
	got, err := linearFloat32([]float32{2, 3}, []float32{1, 0, 0, 1}, tensor)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 3}
	if !closeFloats(got, want) {
		t.Fatalf("linear = %v, want %v", got, want)
	}
}

func TestHFOutputLogitsUseVocabRows(t *testing.T) {
	tensor := modelinfo.SafetensorsTensor{Name: "lm_head.weight", Shape: []uint64{3, 2}}
	got, err := logitsFromOutputWeightFloat32([]float32{1, 2}, []float32{1, 0, 0, 1, 1, 1}, tensor)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3}
	if !closeFloats(got, want) {
		t.Fatalf("logits = %v, want %v", got, want)
	}
}

func TestHFMLP(t *testing.T) {
	identity := modelinfo.SafetensorsTensor{Name: "identity.weight", Shape: []uint64{2, 2}}
	weights := MLPWeights{
		Gate: TensorValues{Values: []float32{1, 0, 0, 1}, Tensor: identity},
		Up:   TensorValues{Values: []float32{1, 0, 0, 1}, Tensor: identity},
		Down: TensorValues{Values: []float32{1, 0, 0, 1}, Tensor: identity},
	}
	got, err := mlpFloat32([]float32{1, 2}, weights)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{silu(1) * 1, silu(2) * 2}
	if !closeFloats(got, want) {
		t.Fatalf("mlp = %v, want %v", got, want)
	}
}

func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func closeFloats(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			return false
		}
	}
	return true
}
