package adapters

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

const CurrentVersion = 1

type TokenBiasAdapter struct {
	Version   int             `json:"version"`
	Type      string          `json:"type"`
	BaseModel string          `json:"base_model,omitempty"`
	Method    string          `json:"method,omitempty"`
	VocabSize int             `json:"vocab_size,omitempty"`
	Bias      map[int]float32 `json:"bias"`
}

func NewTokenBias(baseModel, method string, vocabSize int, counts map[int]int, scale float64) *TokenBiasAdapter {
	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		scale = 0.1
	}
	bias := make(map[int]float32, len(counts))
	for id, count := range counts {
		if id < 0 || count <= 0 {
			continue
		}
		if vocabSize > 0 && id >= vocabSize {
			continue
		}
		bias[id] = float32(math.Log1p(float64(count)) * scale)
	}
	return &TokenBiasAdapter{
		Version:   CurrentVersion,
		Type:      "token_bias",
		BaseModel: baseModel,
		Method:    method,
		VocabSize: vocabSize,
		Bias:      bias,
	}
}

func Load(path string) (*TokenBiasAdapter, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var adapter TokenBiasAdapter
	if err := json.Unmarshal(data, &adapter); err != nil {
		return nil, err
	}
	if err := adapter.Validate(); err != nil {
		return nil, err
	}
	return &adapter, nil
}

func Save(path string, adapter *TokenBiasAdapter) error {
	if path == "" {
		return fmt.Errorf("adapter path is required")
	}
	if adapter == nil {
		return fmt.Errorf("adapter is nil")
	}
	if err := adapter.Validate(); err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(adapter, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

func (a *TokenBiasAdapter) Validate() error {
	if a == nil {
		return fmt.Errorf("adapter is nil")
	}
	if a.Version != CurrentVersion {
		return fmt.Errorf("unsupported adapter version %d", a.Version)
	}
	if a.Type != "token_bias" {
		return fmt.Errorf("unsupported adapter type %q", a.Type)
	}
	for id, value := range a.Bias {
		if id < 0 {
			return fmt.Errorf("adapter token id %d is invalid", id)
		}
		if a.VocabSize > 0 && id >= a.VocabSize {
			return fmt.Errorf("adapter token id %d exceeds vocab size %d", id, a.VocabSize)
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("adapter token id %d has invalid bias", id)
		}
	}
	return nil
}

func (a *TokenBiasAdapter) Apply(logits []float32) []float32 {
	if a == nil || len(a.Bias) == 0 || len(logits) == 0 {
		return logits
	}
	out := append([]float32(nil), logits...)
	for id, bias := range a.Bias {
		if id >= 0 && id < len(out) {
			out[id] += bias
		}
	}
	return out
}

func (a *TokenBiasAdapter) TokenIDs() []int {
	if a == nil {
		return nil
	}
	ids := make([]int, 0, len(a.Bias))
	for id := range a.Bias {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}
