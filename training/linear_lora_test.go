package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/gakon/nego-ai/adapters"
)

func TestLinearLoRACrossEntropy(t *testing.T) {
	for _, logits := range [][]float32{{0, 0}, {10000, 10000}, {-10000, -10000}} {
		loss, gradient, err := linearCrossEntropy(logits, 1)
		if err != nil || math.Abs(loss-math.Log(2)) > 1e-12 || !reflect.DeepEqual(gradient, []float32{.5, -.5}) {
			t.Fatalf("loss=%g gradient=%v err=%v", loss, gradient, err)
		}
	}
	for _, logits := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, _, err := linearCrossEntropy(logits, 0); err == nil {
			t.Fatal("expected invalid loss error")
		}
	}
	g := adapters.LinearLoRAGradients{A: []float32{3}, B: []float32{4}}
	clipLinearGradient(&g, 2.5)
	if g.A[0] != 1.5 || g.B[0] != 2 {
		t.Fatalf("incorrect clipped gradient: %+v", g)
	}
}

func TestFitLinearLoRA(t *testing.T) {
	l, _ := adapters.NewLinearLoRA(2, 2, 2, 2, 7)
	samples := []LinearLoRASample{
		{Input: []float32{1, 0}, BaseLogits: []float32{0, 0}, Target: 0},
		{Input: []float32{0, 1}, BaseLogits: []float32{0, 0}, Target: 1},
	}
	progress := 0
	result, err := FitLinearLoRA(context.Background(), l, samples, LinearLoRAOptions{Epochs: 100, LearningRate: .1, MaxGradNorm: 1, Progress: func(p LinearLoRAProgress) {
		progress++
		if p.Epoch != progress || p.Steps != 2*progress || !finiteLinearLoss(p.MeanLoss) {
			t.Fatalf("invalid progress: %+v", p)
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Steps != 200 || progress != 100 || result.FinalLoss >= result.InitialLoss*.1 {
		t.Fatalf("did not learn: %+v", result)
	}
	before := append([]float32(nil), l.A...)
	if _, err := EvaluateLinearLoRA(context.Background(), l, samples); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, l.A) {
		t.Fatal("evaluation mutated adapter")
	}
}

func TestFitLinearLoRAPreflightAndCancellation(t *testing.T) {
	l, _ := adapters.NewLinearLoRA(2, 2, 1, 1, 1)
	before := append([]float32(nil), l.B...)
	good := LinearLoRASample{Input: []float32{1, 0}, BaseLogits: []float32{0, 0}, Target: 1}
	bad := good
	bad.Target = 2
	opts := LinearLoRAOptions{Epochs: 2, LearningRate: .1}
	if _, err := FitLinearLoRA(context.Background(), l, []LinearLoRASample{good, bad}, opts); err == nil {
		t.Fatal("accepted invalid later sample")
	}
	if !reflect.DeepEqual(before, l.B) {
		t.Fatal("preflight failure changed weights")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := FitLinearLoRA(ctx, l, []LinearLoRASample{good}, opts); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	opts.Progress = func(LinearLoRAProgress) { cancel() }
	result, err := FitLinearLoRA(ctx, l, []LinearLoRASample{good}, opts)
	if !errors.Is(err, context.Canceled) || result.Steps != 1 {
		t.Fatalf("mid-training cancellation: %+v, %v", result, err)
	}
}
