package embeddings

import "testing"

func TestCosine(t *testing.T) {
	got, err := Cosine([]float64{1, 0}, []float64{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("cosine = %f", got)
	}
}

func TestCosineRejectsDifferentDimensions(t *testing.T) {
	if _, err := Cosine([]float64{1}, []float64{1, 2}); err == nil {
		t.Fatal("expected dimension error")
	}
}
