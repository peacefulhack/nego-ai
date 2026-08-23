package native

import (
	"strings"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestEmbeddingLookupFloat32(t *testing.T) {
	tensor := modelinfo.GGUFTensor{Name: "token_embd.weight", Shape: []uint64{3, 2}}
	values := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	got, err := embeddingLookupFloat32(values, tensor, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{4, 5, 6})
}

func TestLogitsFromOutputWeightFloat32(t *testing.T) {
	tensor := modelinfo.GGUFTensor{Name: "output.weight", Shape: []uint64{3, 2}}
	values := []float32{
		1, 0, 0,
		0, 2, 0,
	}
	got, err := logitsFromOutputWeightFloat32([]float32{3, 4, 5}, values, tensor)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{3, 8})
}

func TestEmbeddingHelpersRejectInvalidInputs(t *testing.T) {
	tensor := modelinfo.GGUFTensor{Name: "token_embd.weight", Shape: []uint64{3, 2}}
	if _, err := embeddingLookupFloat32([]float32{1}, tensor, 2, 3); err == nil || !strings.Contains(err.Error(), "range") {
		t.Fatalf("expected range error, got %v", err)
	}
	if _, err := embeddingLookupFloat32([]float32{1}, tensor, 0, 4); err == nil || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("expected dim error, got %v", err)
	}
	if _, err := logitsFromOutputWeightFloat32(nil, []float32{1}, tensor); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("expected empty hidden error, got %v", err)
	}
	if _, err := logitsFromOutputWeightFloat32([]float32{1, 2, 3}, []float32{1}, tensor); err == nil || !strings.Contains(err.Error(), "values") {
		t.Fatalf("expected values error, got %v", err)
	}
}
