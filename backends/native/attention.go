package native

import (
	"fmt"
	"math"
)

type KVCache struct {
	layers []kvLayerCache
	dim    int
}

type kvLayerCache struct {
	keys   [][]float32
	values [][]float32
}

func NewKVCache(layers int, dim int) (*KVCache, error) {
	if layers <= 0 {
		return nil, fmt.Errorf("kv cache layer count must be positive")
	}
	if dim <= 0 {
		return nil, fmt.Errorf("kv cache dimension must be positive")
	}
	return &KVCache{layers: make([]kvLayerCache, layers), dim: dim}, nil
}

func newKVCache(layers int, dim int) (*KVCache, error) {
	return NewKVCache(layers, dim)
}

func (c *KVCache) Append(layer int, key, value []float32) error {
	if c == nil {
		return fmt.Errorf("kv cache is nil")
	}
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("kv cache layer %d is out of range", layer)
	}
	if len(key) != c.dim || len(value) != c.dim {
		return fmt.Errorf("kv cache vector length mismatch: key=%d value=%d want=%d", len(key), len(value), c.dim)
	}
	if err := validateFiniteVector("kv cache key", key); err != nil {
		return err
	}
	if err := validateFiniteVector("kv cache value", value); err != nil {
		return err
	}
	keyCopy := append([]float32(nil), key...)
	valueCopy := append([]float32(nil), value...)
	c.layers[layer].keys = append(c.layers[layer].keys, keyCopy)
	c.layers[layer].values = append(c.layers[layer].values, valueCopy)
	return nil
}

func (c *KVCache) Layer(layer int) ([][]float32, [][]float32, error) {
	if c == nil {
		return nil, nil, fmt.Errorf("kv cache is nil")
	}
	if layer < 0 || layer >= len(c.layers) {
		return nil, nil, fmt.Errorf("kv cache layer %d is out of range", layer)
	}
	return c.layers[layer].keys, c.layers[layer].values, nil
}

func attentionFloat32(query []float32, keys, values [][]float32) ([]float32, error) {
	if len(query) == 0 {
		return nil, fmt.Errorf("attention query is empty")
	}
	if len(keys) == 0 || len(keys) != len(values) {
		return nil, fmt.Errorf("attention key/value length mismatch: keys=%d values=%d", len(keys), len(values))
	}
	scores := make([]float32, len(keys))
	scale := float32(1 / math.Sqrt(float64(len(query))))
	for i := range keys {
		if len(keys[i]) != len(query) || len(values[i]) != len(query) {
			return nil, fmt.Errorf("attention vector length mismatch at %d", i)
		}
		score, err := dotFloat32(query, keys[i])
		if err != nil {
			return nil, err
		}
		scores[i] = score * scale
	}
	probs, err := softmaxFloat32(scores)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(query))
	for i, prob := range probs {
		for j, value := range values[i] {
			out[j] += prob * value
		}
	}
	return out, nil
}

func validateFiniteVector(name string, values []float32) error {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s contains non-finite value", name)
		}
	}
	return nil
}
