package native

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

func multiHeadAttentionFloat32(input []float32, qValues, kValues, vValues, outValues []float32, qTensor, kTensor, vTensor, outTensor modelinfo.GGUFTensor, spec ModelSpec, position int) ([]float32, error) {
	return multiHeadAttentionWithCacheFloat32(input, qValues, kValues, vValues, outValues, qTensor, kTensor, vTensor, outTensor, spec, -1, position, nil)
}

func multiHeadAttentionWithCacheFloat32(input []float32, qValues, kValues, vValues, outValues []float32, qTensor, kTensor, vTensor, outTensor modelinfo.GGUFTensor, spec ModelSpec, layer int, position int, cache *KVCache) ([]float32, error) {
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
	rotatedKHeads := make([][]float32, len(kHeads))
	for i, kHead := range kHeads {
		rotatedK, err := applyRoPEFloat32(kHead, position, spec.RopeTheta)
		if err != nil {
			return nil, fmt.Errorf("k rope head %d: %w", i, err)
		}
		rotatedKHeads[i] = rotatedK
	}
	if cache != nil {
		keyVector, err := concatHeadVectors(rotatedKHeads)
		if err != nil {
			return nil, err
		}
		valueVector, err := concatHeadVectors(vHeads)
		if err != nil {
			return nil, err
		}
		if err := cache.Append(layer, keyVector, valueVector); err != nil {
			return nil, err
		}
	}
	headsPerKV := int(spec.AttentionHeadCount / spec.KVHeadCount)
	concat := make([]float32, 0, len(q))
	for i, qHead := range qHeads {
		kvHead := i / headsPerKV
		rotatedQ, err := applyRoPEFloat32(qHead, position, spec.RopeTheta)
		if err != nil {
			return nil, fmt.Errorf("q rope head %d: %w", i, err)
		}
		keys := [][]float32{rotatedKHeads[kvHead]}
		values := [][]float32{vHeads[kvHead]}
		if cache != nil {
			cacheKeys, cacheValues, err := cache.Layer(layer)
			if err != nil {
				return nil, err
			}
			keys, err = selectCachedKVHead(cacheKeys, kvHead, headDim)
			if err != nil {
				return nil, err
			}
			values, err = selectCachedKVHead(cacheValues, kvHead, headDim)
			if err != nil {
				return nil, err
			}
		}
		headOut, err := attentionFloat32(rotatedQ, keys, values)
		if err != nil {
			return nil, fmt.Errorf("attention head %d: %w", i, err)
		}
		concat = append(concat, headOut...)
	}
	return linearFloat32(concat, outValues, outTensor)
}

func concatHeadVectors(heads [][]float32) ([]float32, error) {
	var total int
	for _, head := range heads {
		if len(head) > int(^uint(0)>>1)-total {
			return nil, fmt.Errorf("head vector length overflows this runtime")
		}
		total += len(head)
	}
	out := make([]float32, 0, total)
	for _, head := range heads {
		out = append(out, head...)
	}
	return out, nil
}

func selectCachedKVHead(vectors [][]float32, kvHead int, headDim int) ([][]float32, error) {
	if kvHead < 0 || headDim <= 0 {
		return nil, fmt.Errorf("invalid kv head selection")
	}
	start := kvHead * headDim
	end := start + headDim
	out := make([][]float32, len(vectors))
	for i, vector := range vectors {
		if end > len(vector) {
			return nil, fmt.Errorf("cached kv vector %d is too short for head %d", i, kvHead)
		}
		out[i] = vector[start:end]
	}
	return out, nil
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
