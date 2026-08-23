package native

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

func multiHeadAttentionFloat32(input []float32, qValues, kValues, vValues, outValues []float32, qTensor, kTensor, vTensor, outTensor modelinfo.GGUFTensor, spec ModelSpec, position int) ([]float32, error) {
	if spec.EmbeddingLength > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("embedding length %d overflows this runtime", spec.EmbeddingLength)
	}
	if spec.KVHeadCount == 0 || spec.AttentionHeadCount%spec.KVHeadCount != 0 {
		return nil, fmt.Errorf("invalid grouped-query dimensions: heads=%d kv_heads=%d", spec.AttentionHeadCount, spec.KVHeadCount)
	}
	if len(input) != int(spec.EmbeddingLength) {
		return nil, fmt.Errorf("multi-head attention input length %d does not match embedding length %d", len(input), spec.EmbeddingLength)
	}
	headDim, err := attentionHeadDim(spec)
	if err != nil {
		return nil, err
	}
	q, err := linearFloat32(input, qValues, qTensor)
	if err != nil {
		return nil, fmt.Errorf("q projection: %w", err)
	}
	k, err := linearFloat32(input, kValues, kTensor)
	if err != nil {
		return nil, fmt.Errorf("k projection: %w", err)
	}
	v, err := linearFloat32(input, vValues, vTensor)
	if err != nil {
		return nil, fmt.Errorf("v projection: %w", err)
	}
	qHeads, err := splitHeads(q, int(spec.AttentionHeadCount), headDim)
	if err != nil {
		return nil, err
	}
	kHeads, err := splitHeads(k, int(spec.KVHeadCount), headDim)
	if err != nil {
		return nil, err
	}
	vHeads, err := splitHeads(v, int(spec.KVHeadCount), headDim)
	if err != nil {
		return nil, err
	}
	headsPerKV := int(spec.AttentionHeadCount / spec.KVHeadCount)
	concat := make([]float32, 0, len(q))
	for i, qHead := range qHeads {
		kvHead := i / headsPerKV
		rotatedQ, err := applyRoPEFloat32(qHead, position, spec.RopeTheta)
		if err != nil {
			return nil, fmt.Errorf("q rope head %d: %w", i, err)
		}
		rotatedK, err := applyRoPEFloat32(kHeads[kvHead], position, spec.RopeTheta)
		if err != nil {
			return nil, fmt.Errorf("k rope head %d: %w", kvHead, err)
		}
		headOut, err := attentionFloat32(rotatedQ, [][]float32{rotatedK}, [][]float32{vHeads[kvHead]})
		if err != nil {
			return nil, fmt.Errorf("attention head %d: %w", i, err)
		}
		concat = append(concat, headOut...)
	}
	return linearFloat32(concat, outValues, outTensor)
}

func attentionHeadDim(spec ModelSpec) (int, error) {
	if spec.AttentionHeadCount == 0 || spec.EmbeddingLength%spec.AttentionHeadCount != 0 {
		return 0, fmt.Errorf("invalid attention dimensions: embedding=%d heads=%d", spec.EmbeddingLength, spec.AttentionHeadCount)
	}
	headDim := spec.EmbeddingLength / spec.AttentionHeadCount
	if headDim == 0 || headDim > uint64(int(^uint(0)>>1)) {
		return 0, fmt.Errorf("attention head dimension %d is invalid on this runtime", headDim)
	}
	return int(headDim), nil
}

func splitHeads(values []float32, heads int, headDim int) ([][]float32, error) {
	if heads <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("heads and head dimension must be positive")
	}
	if heads > int(^uint(0)>>1)/headDim {
		return nil, fmt.Errorf("head shape overflows this runtime")
	}
	if len(values) != heads*headDim {
		return nil, fmt.Errorf("head split length mismatch: values=%d want=%d", len(values), heads*headDim)
	}
	out := make([][]float32, heads)
	for i := 0; i < heads; i++ {
		start := i * headDim
		out[i] = values[start : start+headDim]
	}
	return out, nil
}
