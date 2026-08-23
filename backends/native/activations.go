package native

import (
	"fmt"
	"math"
)

func rmsNormFloat32(x, weight []float32, eps float32) ([]float32, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("rmsnorm input is empty")
	}
	if len(x) != len(weight) {
		return nil, fmt.Errorf("rmsnorm length mismatch: %d != %d", len(x), len(weight))
	}
	if eps < 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return nil, fmt.Errorf("rmsnorm eps must be finite and non-negative")
	}
	var sumSquares float64
	for i, value := range x {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("rmsnorm input contains non-finite value")
		}
		if math.IsNaN(float64(weight[i])) || math.IsInf(float64(weight[i]), 0) {
			return nil, fmt.Errorf("rmsnorm weight contains non-finite value")
		}
		sumSquares += float64(value) * float64(value)
	}
	scale := float32(1 / math.Sqrt(sumSquares/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * scale * weight[i]
	}
	return out, nil
}

func siluFloat32(x []float32) ([]float32, error) {
	out := make([]float32, len(x))
	for i, value := range x {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("silu input contains non-finite value")
		}
		out[i] = value / (1 + float32(math.Exp(float64(-value))))
	}
	return out, nil
}

func softmaxFloat32(x []float32) ([]float32, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("softmax input is empty")
	}
	if err := validateLogits(x); err != nil {
		return nil, err
	}
	maxValue := x[argmax(x)]
	out := make([]float32, len(x))
	var total float64
	for i, value := range x {
		prob := math.Exp(float64(value - maxValue))
		out[i] = float32(prob)
		total += prob
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / total)
	}
	return out, nil
}
