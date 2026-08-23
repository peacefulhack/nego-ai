package native

import (
	"context"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
)

func TestNewDecodeStateCreatesKVCache(t *testing.T) {
	state, err := NewDecodeState(ModelSpec{
		EmbeddingLength:    4,
		BlockCount:         2,
		AttentionHeadCount: 2,
		KVHeadCount:        1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if state.Position != 0 {
		t.Fatalf("unexpected position: %d", state.Position)
	}
	keys, values, err := state.Cache.Layer(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 0 || len(values) != 0 {
		t.Fatalf("expected empty cache, got keys=%d values=%d", len(keys), len(values))
	}
}

func TestForwardTokenWithStateAdvancesAndCaches(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeBlockGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel := model.(*Model)
	state, err := NewDecodeState(nativeModel.Spec())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nativeModel.ForwardTokenWithState(0, state); err != nil {
		t.Fatal(err)
	}
	if state.Position != 1 {
		t.Fatalf("position = %d, want 1", state.Position)
	}
	keys, values, err := state.Cache.Layer(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 1 || len(values) != 1 {
		t.Fatalf("unexpected cache length: keys=%d values=%d", len(keys), len(values))
	}
	if _, err := nativeModel.ForwardTokenWithState(1, state); err != nil {
		t.Fatal(err)
	}
	keys, values, err = state.Cache.Layer(0)
	if err != nil {
		t.Fatal(err)
	}
	if state.Position != 2 || len(keys) != 2 || len(values) != 2 {
		t.Fatalf("unexpected decode state: pos=%d keys=%d values=%d", state.Position, len(keys), len(values))
	}
}

func TestForwardTokenWithStateRejectsNilState(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeBlockGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel := model.(*Model)
	_, err = nativeModel.ForwardTokenWithState(0, nil)
	if err == nil || !strings.Contains(err.Error(), "decode state") {
		t.Fatalf("unexpected error: %v", err)
	}
}
