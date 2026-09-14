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

func TestMultiHeadAttentionWithCacheAppendsKV(t *testing.T) {
	spec := ModelSpec{
		EmbeddingLength:    4,
		AttentionHeadCount: 2,
		KVHeadCount:        1,
		RopeTheta:          10000,
	}
	cache, err := NewKVCache(1, 2)
	if err != nil {
		t.Fatal(err)
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
	if _, err := multiHeadAttentionWithCacheFloat32(
		[]float32{1, 2, 3, 4},
		AttentionWeights{
			QValues:   identity4,
			KValues:   firstTwo,
			VValues:   firstTwo,
			OutValues: identity4,
			QTensor:   qTensor,
			KTensor:   kTensor,
			VTensor:   vTensor,
			OutTensor: outTensor,
		},
		spec,
		0,
		0,
		cache,
	); err != nil {
		t.Fatal(err)
	}
	keys, values, err := cache.Layer(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 1 || len(values) != 1 {
		t.Fatalf("unexpected cache length: keys=%d values=%d", len(keys), len(values))
	}
	assertFloat32Slice(t, keys[0], []float32{1, 2})
	assertFloat32Slice(t, values[0], []float32{1, 2})
}

func TestMultiHeadAttentionUsesQKNorms(t *testing.T) {
	spec := ModelSpec{
		EmbeddingLength:    4,
		AttentionHeadCount: 2,
		KVHeadCount:        1,
		RopeTheta:          10000,
		RMSNormEpsilon:     0,
	}
	tensor4x4 := modelinfo.GGUFTensor{Name: "q", Shape: []uint64{4, 4}}
	tensor4x2 := modelinfo.GGUFTensor{Name: "k", Shape: []uint64{4, 2}}
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
	_, err := multiHeadAttentionWithCacheFloat32(
		[]float32{1, 2, 3, 4},
		AttentionWeights{
			QValues:   identity4,
			QNorm:     []float32{1},
			KValues:   firstTwo,
			KNorm:     []float32{1, 1},
			VValues:   firstTwo,
			OutValues: identity4,
			QTensor:   tensor4x4,
			KTensor:   tensor4x2,
			VTensor:   tensor4x2,
			OutTensor: tensor4x4,
		},
		spec,
		0,
		0,
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "q norm head") {
		t.Fatalf("expected q norm error, got %v", err)
	}
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
