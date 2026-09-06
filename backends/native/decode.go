package native

import "fmt"

type DecodeState struct {
	Position int
	Cache    *KVCache
}

func NewDecodeState(spec ModelSpec) (*DecodeState, error) {
	if spec.BlockCount > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("block count %d overflows this runtime", spec.BlockCount)
	}
	if spec.KVHeadCount > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("kv head count %d overflows this runtime", spec.KVHeadCount)
	}
	headDim, err := attentionHeadDim(spec)
	if err != nil {
		return nil, err
	}
	kvHeads := int(spec.KVHeadCount)
	if kvHeads <= 0 {
		return nil, fmt.Errorf("kv head count must be positive")
	}
	if kvHeads > int(^uint(0)>>1)/headDim {
		return nil, fmt.Errorf("kv cache dimension overflows this runtime")
	}
	cache, err := NewKVCache(int(spec.BlockCount), kvHeads*headDim)
	if err != nil {
		return nil, err
	}
	return &DecodeState{Cache: cache}, nil
}

func (s *DecodeState) advance() error {
	if s == nil {
		return fmt.Errorf("decode state is nil")
	}
	if s.Position == int(^uint(0)>>1) {
		return fmt.Errorf("decode position overflows this runtime")
	}
	s.Position++
	return nil
}
