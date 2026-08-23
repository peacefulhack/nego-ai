package native

import (
	"math"
	"strings"
	"testing"
)

func TestKVCacheAppendAndLayer(t *testing.T) {
	cache, err := NewKVCache(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	key := []float32{1, 0}
	value := []float32{0, 1}
	if err := cache.Append(1, key, value); err != nil {
		t.Fatal(err)
	}
	key[0] = 9
	keys, values, err := cache.Layer(1)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, keys[0], []float32{1, 0})
	assertFloat32Slice(t, values[0], []float32{0, 1})
}

func TestAttentionFloat32(t *testing.T) {
	got, err := attentionFloat32(
		[]float32{1, 0},
		[][]float32{{10, 0}, {0, 10}},
		[][]float32{{2, 0}, {0, 3}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] <= 1.99 || got[1] >= 0.01 {
		t.Fatalf("unexpected attention output: %#v", got)
	}
}

func TestAttentionRejectsInvalidInputs(t *testing.T) {
	if _, err := NewKVCache(0, 1); err == nil {
		t.Fatal("expected invalid cache layer count error")
	}
	cache, err := NewKVCache(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append(0, []float32{1}, []float32{1}); err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("expected append mismatch error, got %v", err)
	}
	if err := cache.Append(0, []float32{float32(math.NaN()), 0}, []float32{1, 0}); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("expected append finite error, got %v", err)
	}
	if _, err := attentionFloat32(nil, nil, nil); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("expected empty query error, got %v", err)
	}
	if _, err := attentionFloat32([]float32{1}, [][]float32{{1}}, nil); err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("expected kv mismatch error, got %v", err)
	}
	if _, err := attentionFloat32([]float32{1, 2}, [][]float32{{1}}, [][]float32{{1}}); err == nil || !strings.Contains(err.Error(), "length") {
		t.Fatalf("expected length error, got %v", err)
	}
}
