package embeddings

import (
	"fmt"
	"math"
)

func Cosine(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("embedding dimensions differ: %d != %d", len(a), len(b))
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0, fmt.Errorf("cannot compare zero vector")
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}
