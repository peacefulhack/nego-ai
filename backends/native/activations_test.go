package native

import (
	"math"
	"testing"
)

func TestRMSNormFloat32(t *testing.T) {
	got, err := rmsNormFloat32([]float32{3, 4}, []float32{1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(1 / math.Sqrt(12.5))
	assertFloat32Slice(t, got, []float32{3 * scale, 8 * scale})
}

func TestSiluFloat32(t *testing.T) {
	got, err := siluFloat32([]float32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{0, 1 / (1 + float32(math.Exp(-1)))})
}

func TestSoftmaxFloat32(t *testing.T) {
	got, err := softmaxFloat32([]float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	var total float32
	for _, value := range got {
		total += value
	}
	if math.Abs(float64(total-1)) > 1e-6 {
		t.Fatalf("softmax total = %v", total)
	}
	if !(got[2] > got[1] && got[1] > got[0]) {
		t.Fatalf("unexpected softmax order: %#v", got)
	}
}

func TestActivationRejectsInvalidInputs(t *testing.T) {
	if _, err := rmsNormFloat32(nil, nil, 0); err == nil {
		t.Fatal("expected empty rmsnorm error")
	}
	if _, err := rmsNormFloat32([]float32{1}, []float32{1}, -1); err == nil {
		t.Fatal("expected invalid eps error")
	}
	if _, err := siluFloat32([]float32{float32(math.Inf(1))}); err == nil {
		t.Fatal("expected invalid silu input error")
	}
	if _, err := softmaxFloat32(nil); err == nil {
		t.Fatal("expected empty softmax error")
	}
}
