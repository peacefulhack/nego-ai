package native

import (
	"math"
	"testing"
)

func TestAddAndMulFloat32(t *testing.T) {
	added, err := addFloat32([]float32{1, 2}, []float32{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, added, []float32{4, 6})

	multiplied, err := mulFloat32([]float32{1, 2}, []float32{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, multiplied, []float32{3, 8})
}

func TestApplyRoPEFloat32(t *testing.T) {
	unchanged, err := applyRoPEFloat32([]float32{1, 0, 0, 1}, 0, 10000)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, unchanged, []float32{1, 0, 0, 1})

	rotated, err := applyRoPEFloat32([]float32{1, 0}, 1, 10000)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, rotated, []float32{float32(math.Cos(1)), float32(math.Sin(1))})
}

func TestVectorHelpersRejectInvalidInputs(t *testing.T) {
	if _, err := addFloat32([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("expected add length error")
	}
	if _, err := mulFloat32([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("expected mul length error")
	}
	if _, err := applyRoPEFloat32([]float32{1}, 0, 10000); err == nil {
		t.Fatal("expected odd rope length error")
	}
	if _, err := applyRoPEFloat32([]float32{1, 0}, -1, 10000); err == nil {
		t.Fatal("expected negative position error")
	}
	if _, err := applyRoPEFloat32([]float32{1, 0}, 0, 0); err == nil {
		t.Fatal("expected invalid theta error")
	}
	if _, err := addFloat32([]float32{float32(math.NaN())}, []float32{1}); err == nil {
		t.Fatal("expected invalid add input error")
	}
	if _, err := applyRoPEFloat32([]float32{float32(math.Inf(1)), 0}, 0, 10000); err == nil {
		t.Fatal("expected invalid rope input error")
	}
}
