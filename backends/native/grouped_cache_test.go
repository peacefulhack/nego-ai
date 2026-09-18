package native

import (
	"fmt"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func groupedAttentionFixture(heads, kvHeads, headDim int) (ModelSpec, AttentionWeights, []float32) {
	embed, kvDim := heads*headDim, kvHeads*headDim
	projection := func(rows int) []float32 {
		values := make([]float32, rows*embed)
		for i := 0; i < rows; i++ {
			values[i*embed+i] = 1
		}
		return values
	}
	weights := AttentionWeights{
		QValues: projection(embed), KValues: projection(kvDim), VValues: projection(kvDim), OutValues: projection(embed),
		QTensor:   modelinfo.GGUFTensor{Shape: []uint64{uint64(embed), uint64(embed)}},
		KTensor:   modelinfo.GGUFTensor{Shape: []uint64{uint64(embed), uint64(kvDim)}},
		VTensor:   modelinfo.GGUFTensor{Shape: []uint64{uint64(embed), uint64(kvDim)}},
		OutTensor: modelinfo.GGUFTensor{Shape: []uint64{uint64(embed), uint64(embed)}},
	}
	input := make([]float32, embed)
	for i := range input {
		input[i] = float32(i%7+1) / 8
	}
	spec := ModelSpec{Architecture: "qwen3", EmbeddingLength: uint64(embed), AttentionHeadCount: uint64(heads), KVHeadCount: uint64(kvHeads), RopeTheta: 10000}
	return spec, weights, input
}

func BenchmarkCachedGroupedAttention(b *testing.B) {
	for _, tokens := range []int{128, 1024} {
		b.Run(fmt.Sprintf("tokens_%d", tokens), func(b *testing.B) {
			spec, weights, input := groupedAttentionFixture(8, 2, 16)
			cache, err := NewKVCache(1, 32)
			if err != nil {
				b.Fatal(err)
			}
			for i := 0; i < tokens; i++ {
				if err := cache.Append(0, input[:32], input[:32]); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := multiHeadAttentionWithCacheFloat32(input, weights, spec, 0, tokens, cache)
				if err != nil {
					b.Fatal(err)
				}
				// Keep a fixed history length while retaining backing-slice capacity.
				cache.layers[0].keys = cache.layers[0].keys[:tokens]
				cache.layers[0].values = cache.layers[0].values[:tokens]
			}
		})
	}
}

func TestCachedGroupedAttentionMatchesPerHeadReference(t *testing.T) {
	for _, kvHeads := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("kv_heads_%d", kvHeads), func(t *testing.T) {
			spec, weights, input := groupedAttentionFixture(4, kvHeads, 4)
			cache, err := NewKVCache(1, kvHeads*4)
			if err != nil {
				t.Fatal(err)
			}
			for position := 0; position < 3; position++ {
				for i := range input {
					input[i] += float32(position) / 16
				}
				got, err := multiHeadAttentionWithCacheFloat32(input, weights, spec, 0, position, cache)
				if err != nil {
					t.Fatal(err)
				}
				keys, values, err := cache.Layer(0)
				if err != nil {
					t.Fatal(err)
				}
				var want []float32
				for head := 0; head < 4; head++ {
					kvHead := head / (4 / kvHeads)
					query, err := spec.applyRoPE(input[head*4:head*4+4], position)
					if err != nil {
						t.Fatal(err)
					}
					k, err := selectCachedKVHead(keys, kvHead, 4)
					if err != nil {
						t.Fatal(err)
					}
					v, err := selectCachedKVHead(values, kvHead, 4)
					if err != nil {
						t.Fatal(err)
					}
					out, err := attentionFloat32(query, k, v)
					if err != nil {
						t.Fatal(err)
					}
					want = append(want, out...)
				}
				assertFloat32Slice(t, got, want)
			}
		})
	}
}

func TestSelectCachedKVHeadRejectsInvalidRanges(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct{ head, dim int }{
		{-1, 2}, {0, 0}, {0, -1}, {maxInt, 1}, {maxInt / 2, 2}, {1, maxInt}, {1, 2},
	} {
		if _, err := selectCachedKVHead([][]float32{{1, 2}}, tc.head, tc.dim); err == nil {
			t.Fatalf("expected invalid range error for %+v", tc)
		}
	}
	if _, err := selectCachedKVHead([][]float32{{1, 2}, {3}}, 0, 2); err == nil {
		t.Fatal("expected short later vector error")
	}
}

func TestUncachedGroupedAttentionUsesEachKVHead(t *testing.T) {
	spec, weights, input := groupedAttentionFixture(4, 2, 4)
	got, err := multiHeadAttentionWithCacheFloat32(input, weights, spec, -1, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	var want []float32
	for head := 0; head < 4; head++ {
		start := (head / 2) * 4
		want = append(want, input[start:start+4]...)
	}
	assertFloat32Slice(t, got, want)
}
