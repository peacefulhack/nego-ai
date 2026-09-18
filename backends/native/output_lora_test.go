package native

import (
	"context"
	"path/filepath"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/training"
)

func TestGGUFTrainReloadOutputLoRA(t *testing.T) {
	ctx := context.Background()
	path := fakeBlockGGUF(t)
	loaded, err := Backend{}.Load(ctx, nego.ModelOptions{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	m := loaded.(*Model)
	state, err := NewDecodeState(m.spec)
	if err != nil {
		t.Fatal(err)
	}
	hidden, base, err := m.ForwardFeaturesWithState(0, state)
	if err != nil || state.Position != 1 {
		t.Fatalf("features failed: %v", err)
	}
	plain, err := m.ForwardToken(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, base, plain)
	l, err := adapters.NewLinearLoRA(len(hidden), len(base), 1, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	result, err := training.FitLinearLoRA(ctx, l, []training.LinearLoRASample{{Input: hidden, BaseLogits: base, Target: 1}}, training.LinearLoRAOptions{Epochs: 50, LearningRate: .1, MaxGradNorm: 1})
	if err != nil || result.FinalLoss >= result.InitialLoss {
		t.Fatalf("training failed: %+v, %v", result, err)
	}
	unchanged, err := m.ForwardToken(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, unchanged, base)
	checkpoint := filepath.Join(t.TempDir(), "head.json")
	if err := adapters.SaveLinearLoRA(checkpoint, l); err != nil {
		t.Fatal(err)
	}
	adapted, err := Backend{}.Load(ctx, nego.ModelOptions{Path: path, Options: map[string]string{"output_lora": checkpoint}})
	if err != nil {
		t.Fatal(err)
	}
	defer adapted.Close()
	got, err := adapted.(*Model).ForwardToken(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := l.Forward(hidden, base)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, want)
	state, _ = NewDecodeState(m.spec)
	_, frozen, err := adapted.(*Model).ForwardFeaturesWithState(0, state)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, frozen, base)
	out, err := adapted.Generate(ctx, nego.GenerateRequest{Prompt: "hello", MaxTokens: 1})
	if err != nil || out.Text != " done" {
		t.Fatalf("adapted generation = %v, %v", out, err)
	}
	if !adapted.(*Model).RuntimeStats().AdapterLoaded {
		t.Fatal("missing adapter runtime status")
	}
}

func TestGGUFFrozenFeaturesRejectInvalidState(t *testing.T) {
	loaded, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeBlockGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	m := loaded.(*Model)
	if _, _, err := m.ForwardFeaturesWithState(0, nil); err == nil {
		t.Fatal("accepted nil state")
	}
	state, err := NewDecodeState(m.spec)
	if err != nil {
		t.Fatal(err)
	}
	state.Position = int(m.spec.ContextLength)
	if _, _, err := m.ForwardFeaturesWithState(0, state); err == nil {
		t.Fatal("accepted context overflow")
	}
}

func TestGGUFOutputLoRARejectsWrongDimensions(t *testing.T) {
	l, _ := adapters.NewLinearLoRA(3, 6, 1, 1, 1)
	path := filepath.Join(t.TempDir(), "head.json")
	if err := adapters.SaveLinearLoRA(path, l); err != nil {
		t.Fatal(err)
	}
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeBlockGGUF(t), Options: map[string]string{"output_lora": path}})
	if err == nil {
		model.Close()
		t.Fatal("accepted mismatched adapter")
	}
}
