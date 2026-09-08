package nativehf

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

type DecodeState struct {
	Position int
	layers   []decodeLayerCache
}

type decodeLayerCache struct {
	keys   [][][]float32
	values [][][]float32
}

func NewDecodeState(spec modelinfo.HFModelSpec) (*DecodeState, error) {
	if spec.BlockCount == 0 {
		return nil, fmt.Errorf("native-hf block count must be positive")
	}
	if spec.KVHeadCount == 0 {
		return nil, fmt.Errorf("native-hf kv head count must be positive")
	}
	if spec.BlockCount > uint64(int(^uint(0)>>1)) || spec.KVHeadCount > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("native-hf decode dimensions overflow this runtime")
	}
	state := &DecodeState{layers: make([]decodeLayerCache, int(spec.BlockCount))}
	kvHeads := int(spec.KVHeadCount)
	for i := range state.layers {
		state.layers[i] = decodeLayerCache{
			keys:   make([][][]float32, kvHeads),
			values: make([][][]float32, kvHeads),
		}
	}
	return state, nil
}

func (s *DecodeState) Append(layer, kvHead int, key, value []float32) error {
	cache, err := s.layer(layer, kvHead)
	if err != nil {
		return err
	}
	if len(key) == 0 || len(value) == 0 || len(key) != len(value) {
		return fmt.Errorf("native-hf kv cache vector length mismatch: key=%d value=%d", len(key), len(value))
	}
	cache.keys[kvHead] = append(cache.keys[kvHead], append([]float32(nil), key...))
	cache.values[kvHead] = append(cache.values[kvHead], append([]float32(nil), value...))
	return nil
}

func (s *DecodeState) Layer(layer, kvHead int) ([][]float32, [][]float32, error) {
	cache, err := s.layer(layer, kvHead)
	if err != nil {
		return nil, nil, err
	}
	return cache.keys[kvHead], cache.values[kvHead], nil
}

func (s *DecodeState) CachedTokens(layer, kvHead int) int {
	keys, _, err := s.Layer(layer, kvHead)
	if err != nil {
		return 0
	}
	return len(keys)
}

func (s *DecodeState) layer(layer, kvHead int) (*decodeLayerCache, error) {
	if s == nil {
		return nil, fmt.Errorf("native-hf decode state is nil")
	}
	if layer < 0 || layer >= len(s.layers) {
		return nil, fmt.Errorf("native-hf decode layer %d is out of range", layer)
	}
	if kvHead < 0 || kvHead >= len(s.layers[layer].keys) {
		return nil, fmt.Errorf("native-hf decode kv head %d is out of range", kvHead)
	}
	return &s.layers[layer], nil
}

func (s *DecodeState) advance() error {
	if s == nil {
		return fmt.Errorf("native-hf decode state is nil")
	}
	if s.Position == int(^uint(0)>>1) {
		return fmt.Errorf("native-hf decode position overflows this runtime")
	}
	s.Position++
	return nil
}
