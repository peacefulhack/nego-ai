package training

import (
	"context"
	"fmt"
	"math"

	"github.com/gakon/nego-ai/adapters"
)

// LinearLoRASample is one detached frozen-model forward pass paired with its
// next-token target. Never include held-out evaluation samples in training.
type LinearLoRASample struct {
	Input      []float32
	BaseLogits []float32
	Target     int
}

type LinearLoRAOptions struct {
	Epochs       int
	LearningRate float32
	// MaxGradNorm clips the combined A/B gradient L2 norm; zero disables clipping.
	MaxGradNorm float32
	Progress    func(LinearLoRAProgress)
}

type LinearLoRAProgress struct {
	Epoch, Epochs, Steps int
	MeanLoss             float64
}

type LinearLoRAResult struct {
	Steps       int
	InitialLoss float64
	FinalLoss   float64
}

// FitLinearLoRA trains an output-head adapter using mean next-token cross
// entropy and sample-wise SGD over frozen features. Only A/B are updated. The
// adapter is mutated after each valid step; cancellation retains completed steps.
// It does not backpropagate into the transformer or train attention adapters.
func FitLinearLoRA(ctx context.Context, adapter *adapters.LinearLoRA, samples []LinearLoRASample, opts LinearLoRAOptions) (LinearLoRAResult, error) {
	var result LinearLoRAResult
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if err := adapter.Validate(); err != nil {
		return result, err
	}
	if len(samples) == 0 {
		return result, fmt.Errorf("LoRA training samples are empty")
	}
	if opts.Epochs <= 0 || opts.Epochs > int(^uint(0)>>1)/len(samples) {
		return result, fmt.Errorf("invalid LoRA epoch count")
	}
	if !finiteLinearLoss(float64(opts.LearningRate)) || opts.LearningRate <= 0 {
		return result, fmt.Errorf("LoRA learning rate must be finite and positive")
	}
	if !finiteLinearLoss(float64(opts.MaxGradNorm)) || opts.MaxGradNorm < 0 {
		return result, fmt.Errorf("LoRA gradient norm limit must be finite and non-negative")
	}
	// Preflight the entire dataset before the first update.
	for i, sample := range samples {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if len(sample.Input) != adapter.InputSize || len(sample.BaseLogits) != adapter.OutputSize || sample.Target < 0 || sample.Target >= adapter.OutputSize {
			return result, fmt.Errorf("LoRA sample %d has invalid shape or target", i)
		}
		for _, vector := range [][]float32{sample.Input, sample.BaseLogits} {
			for _, value := range vector {
				if !finiteLinearLoss(float64(value)) {
					return result, fmt.Errorf("LoRA sample %d contains a non-finite value", i)
				}
			}
		}
	}
	initial, err := EvaluateLinearLoRA(ctx, adapter, samples)
	if err != nil {
		return result, err
	}
	result.InitialLoss = initial
	for epoch := 0; epoch < opts.Epochs; epoch++ {
		var lossSum float64
		for i, sample := range samples {
			if err := ctx.Err(); err != nil {
				return result, err
			}
			logits, err := adapter.Forward(sample.Input, sample.BaseLogits)
			if err != nil {
				return result, fmt.Errorf("LoRA sample %d forward: %w", i, err)
			}
			loss, dy, err := linearCrossEntropy(logits, sample.Target)
			if err != nil {
				return result, err
			}
			gradient, err := adapter.Backward(sample.Input, dy)
			if err != nil {
				return result, err
			}
			clipLinearGradient(&gradient, opts.MaxGradNorm)
			if err := adapter.StepSGD(gradient, opts.LearningRate); err != nil {
				return result, err
			}
			lossSum += loss
			result.Steps++
		}
		if opts.Progress != nil {
			opts.Progress(LinearLoRAProgress{Epoch: epoch + 1, Epochs: opts.Epochs, Steps: result.Steps, MeanLoss: lossSum / float64(len(samples))})
		}
	}
	result.FinalLoss, err = EvaluateLinearLoRA(ctx, adapter, samples)
	return result, err
}

// EvaluateLinearLoRA returns mean next-token cross entropy without updating the
// adapter. Pass a separate held-out sample set to measure validation loss.
func EvaluateLinearLoRA(ctx context.Context, adapter *adapters.LinearLoRA, samples []LinearLoRASample) (float64, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := adapter.Validate(); err != nil {
		return 0, err
	}
	if len(samples) == 0 {
		return 0, fmt.Errorf("LoRA evaluation samples are empty")
	}
	var mean float64
	for i, sample := range samples {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		logits, err := adapter.Forward(sample.Input, sample.BaseLogits)
		if err != nil {
			return 0, err
		}
		loss, _, err := linearCrossEntropy(logits, sample.Target)
		if err != nil {
			return 0, err
		}
		mean += (loss - mean) / float64(i+1)
	}
	return mean, nil
}

func linearCrossEntropy(logits []float32, target int) (float64, []float32, error) {
	if target < 0 || target >= len(logits) {
		return 0, nil, fmt.Errorf("LoRA target is out of range")
	}
	max := math.Inf(-1)
	for _, value := range logits {
		if !finiteLinearLoss(float64(value)) {
			return 0, nil, fmt.Errorf("LoRA logits must be finite")
		}
		if float64(value) > max {
			max = float64(value)
		}
	}
	var sum float64
	for _, value := range logits {
		sum += math.Exp(float64(value) - max)
	}
	gradient := make([]float32, len(logits))
	for i, value := range logits {
		gradient[i] = float32(math.Exp(float64(value)-max) / sum)
	}
	gradient[target] -= 1
	return (max - float64(logits[target])) + math.Log(sum), gradient, nil
}

func clipLinearGradient(g *adapters.LinearLoRAGradients, limit float32) {
	if limit == 0 {
		return
	}
	var squares float64
	for _, vector := range [][]float32{g.A, g.B} {
		for _, value := range vector {
			squares += float64(value) * float64(value)
		}
	}
	norm := math.Sqrt(squares)
	if norm <= float64(limit) {
		return
	}
	scale := float64(limit) / norm
	for _, vector := range [][]float32{g.A, g.B} {
		for i, value := range vector {
			vector[i] = float32(float64(value) * scale)
		}
	}
}

func finiteLinearLoss(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
