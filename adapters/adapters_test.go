package adapters

import (
	"path/filepath"
	"testing"
)

func TestTokenBiasAdapterSaveLoadApply(t *testing.T) {
	adapter := NewTokenBias("model.gguf", "token-bias", 4, map[int]int{1: 3, 2: 1}, 0.5)
	path := filepath.Join(t.TempDir(), "adapter.json")
	if err := Save(path, adapter); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Type != "token_bias" || len(loaded.Bias) != 2 {
		t.Fatalf("unexpected adapter: %#v", loaded)
	}
	logits := loaded.Apply([]float32{0, 0, 0, 0})
	if logits[1] <= logits[2] || logits[0] != 0 {
		t.Fatalf("unexpected logits: %#v", logits)
	}
	ids := loaded.TokenIDs()
	if len(ids) != 2 || ids[0] != 1 || ids[1] != 2 {
		t.Fatalf("unexpected ids: %#v", ids)
	}
}

func TestTokenBiasAdapterRejectsInvalidToken(t *testing.T) {
	adapter := &TokenBiasAdapter{
		Version:   CurrentVersion,
		Type:      "token_bias",
		VocabSize: 2,
		Bias:      map[int]float32{3: 1},
	}
	if err := adapter.Validate(); err == nil {
		t.Fatal("expected validation error")
	}
}
