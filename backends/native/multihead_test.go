package native

import (
	"strings"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestMultiHeadAttentionFloat32(t *testing.T) {
	spec := ModelSpec{
		EmbeddingLength:    4,
		AttentionHeadCount: 2,
		KVHeadCount:        1,
		RopeTheta:          10000,
	}
	qTensor := modelinfo.GGUFTensor{Name: "q", Shape: []uint64{4, 4}}
	kTensor := modelinfo.GGUFTensor{Name: "k", Shape: []uint64{4, 2}}
	vTensor := modelinfo.GGUFTensor{Name: "v", Shape: []uint64{4, 2}}
	outTensor := modelinfo.GGUFTensor{Name: "o", Shape: []uint64{4, 4}}
	identity4 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	firstTwo := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}
	got, err := multiHeadAttentionFloat32(
		[]float32{1, 2, 3, 4},
		identity4,
		firstTwo,
		firstTwo,
		identity4,
		qTensor,
		kTensor,
		vTensor,
		outTensor,
		spec,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{1, 2, 1, 2})
}

func TestSplitHeads(t *testing.T) {
	heads, err := splitHeads([]float32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, heads[0], []float32{1, 2})
	assertFloat32Slice(t, heads[1], []float32{3, 4})
}

func TestMultiHeadAttentionRejectsInvalidInputs(t *testing.T) {
	spec := ModelSpec{EmbeddingLength: 3, AttentionHeadCount: 2, KVHeadCount: 1, RopeTheta: 10000}
	_, err := multiHeadAttentionFloat32(nil, nil, nil, nil, nil, modelinfo.GGUFTensor{}, modelinfo.GGUFTensor{}, modelinfo.GGUFTensor{}, modelinfo.GGUFTensor{}, spec, 0)
	if err == nil || !strings.Contains(err.Error(), "input length") {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := splitHeads([]float32{1}, 2, 2); err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("unexpected split error: %v", err)
	}
	spec = ModelSpec{EmbeddingLength: 2, AttentionHeadCount: 2, KVHeadCount: 0, RopeTheta: 10000}
	_, err = multiHeadAttentionFloat32([]float32{1, 2}, nil, nil, nil, nil, modelinfo.GGUFTensor{}, modelinfo.GGUFTensor{}, modelinfo.GGUFTensor{}, modelinfo.GGUFTensor{}, spec, 0)
	if err == nil || !strings.Contains(err.Error(), "grouped-query") {
		t.Fatalf("unexpected grouped-query error: %v", err)
	}
}
