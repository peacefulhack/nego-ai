package native

import (
	"math"
	"strings"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestLinearFloat32(t *testing.T) {
	tensor := modelinfo.GGUFTensor{Name: "proj.weight", Shape: []uint64{2, 3}}
	values := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	got, err := linearFloat32([]float32{2, 3}, values, tensor)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{2, 3, 5})
}

func TestMLPFloat32(t *testing.T) {
	gateTensor := modelinfo.GGUFTensor{Name: "ffn_gate.weight", Shape: []uint64{2, 2}}
	upTensor := modelinfo.GGUFTensor{Name: "ffn_up.weight", Shape: []uint64{2, 2}}
	downTensor := modelinfo.GGUFTensor{Name: "ffn_down.weight", Shape: []uint64{2, 2}}
	identity := []float32{
		1, 0,
		0, 1,
	}
	got, err := mlpFloat32([]float32{0, 1}, identity, identity, identity, gateTensor, upTensor, downTensor)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{0, 1 / (1 + float32(math.Exp(-1)))})
}

func TestMLPRejectsInvalidInputs(t *testing.T) {
	tensor := modelinfo.GGUFTensor{Name: "proj.weight", Shape: []uint64{2, 1}}
	if _, err := linearFloat32(nil, []float32{1, 2}, tensor); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("expected empty input error, got %v", err)
	}
	if _, err := linearFloat32([]float32{1, 2}, []float32{1}, tensor); err == nil || !strings.Contains(err.Error(), "values") {
		t.Fatalf("expected value count error, got %v", err)
	}
}
