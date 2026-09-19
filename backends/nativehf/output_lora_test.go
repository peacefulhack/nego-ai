package nativehf

import (
	"context"
	"path/filepath"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/training"
)

func TestHFTrainReloadOutputLoRA(t *testing.T) {
	ctx := context.Background()
	path := writeTinyForwardModel(t)
	loaded, err := Backend{}.Load(ctx, nego.ModelOptions{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	m := loaded.(*Model)
	state, err := NewDecodeState(*m.spec)
	if err != nil {
		t.Fatal(err)
	}
	hidden, base, err := m.ForwardFeaturesWithState(0, state)
	if err != nil || state.Position != 1 {
		t.Fatalf("features: %v", err)
	}
	plain, err := m.ForwardToken(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxFloat32Slice(t, base, plain, 1e-6)
	l, err := adapters.NewLinearLoRA(len(hidden), len(base), 1, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	result, err := training.FitLinearLoRA(ctx, l, []training.LinearLoRASample{{Input: hidden, BaseLogits: base, Target: 1}}, training.LinearLoRAOptions{Epochs: 50, LearningRate: .1, MaxGradNorm: 1})
	if err != nil || result.FinalLoss >= result.InitialLoss {
		t.Fatalf("training: %+v, %v", result, err)
	}
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
	assertApproxFloat32Slice(t, got, want, 1e-6)
	state, _ = NewDecodeState(*m.spec)
	_, frozen, err := adapted.(*Model).ForwardFeaturesWithState(0, state)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxFloat32Slice(t, frozen, base, 1e-6)
	unchanged, err := m.ForwardToken(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxFloat32Slice(t, unchanged, base, 1e-6)
	out, err := adapted.Generate(ctx, nego.GenerateRequest{Prompt: "a", MaxTokens: 1})
	if err != nil || out.Text != "b" {
		t.Fatalf("generation: %v, %v", out, err)
	}
	if !adapted.(*Model).RuntimeStats().AdapterLoaded {
		t.Fatal("missing adapter status")
	}
}

func TestHFOutputLoRARejectsDimensionsAndState(t *testing.T) {
	path := writeTinyForwardModel(t)
	l, _ := adapters.NewLinearLoRA(3, 2, 1, 1, 1)
	checkpoint := filepath.Join(t.TempDir(), "head.json")
	if err := adapters.SaveLinearLoRA(checkpoint, l); err != nil {
		t.Fatal(err)
	}
	if model, err := (Backend{}).Load(context.Background(), nego.ModelOptions{Path: path, Options: map[string]string{"output_lora": checkpoint}}); err == nil {
		model.Close()
		t.Fatal("accepted wrong adapter shape")
	}
	loaded, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	m := loaded.(*Model)
	if _, _, err := m.ForwardFeaturesWithState(0, nil); err == nil {
		t.Fatal("accepted nil state")
	}
	state, _ := NewDecodeState(*m.spec)
	state.Position = int(m.spec.ContextLength)
	if _, _, err := m.ForwardFeaturesWithState(0, state); err == nil {
		t.Fatal("accepted context overflow")
	}
}
