package native

import (
	"fmt"
	"math"
)

func addFloat32(a, b []float32) ([]float32, error) {
	if err := validateVectorPair("add", a, b); err != nil {
		return nil, err
	}
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out, nil
}

func mulFloat32(a, b []float32) ([]float32, error) {
	if err := validateVectorPair("mul", a, b); err != nil {
		return nil, err
	}
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out, nil
}

func applyRoPEFloat32(x []float32, position int, theta float64) ([]float32, error) {
	return applyRoPELayoutFloat32(x, position, theta, false)
}

func (s ModelSpec) applyRoPE(x []float32, position int) ([]float32, error) {
	// GGUF Llama Q/K weights are permuted for adjacent pairs; Qwen retains
	// the split-half (NeoX) layout. See llama_model_rope_type in llama.cpp.
	splitHalf := s.Architecture == "qwen2" || s.Architecture == "qwen3"
	return applyRoPELayoutFloat32(x, position, s.RopeTheta, splitHalf)
}

func applyRoPELayoutFloat32(x []float32, position int, theta float64, splitHalf bool) ([]float32, error) {
	if len(x) == 0 || len(x)%2 != 0 {
		return nil, fmt.Errorf("rope input length must be positive and even")
	}
	if position < 0 {
		return nil, fmt.Errorf("rope position must be non-negative")
	}
	if theta <= 0 || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return nil, fmt.Errorf("rope theta must be finite and positive")
	}
	out := make([]float32, len(x))
	dim := float64(len(x))
	for i := 0; i < len(x); i += 2 {
		left, right := i, i+1
		if splitHalf {
			left, right = i/2, i/2+len(x)/2
		}
		if math.IsNaN(float64(x[left])) || math.IsInf(float64(x[left]), 0) || math.IsNaN(float64(x[right])) || math.IsInf(float64(x[right]), 0) {
			return nil, fmt.Errorf("rope input contains non-finite value")
		}
		freq := math.Pow(theta, -float64(i)/dim)
		angle := float64(position) * freq
		cos, sin := math.Cos(angle), math.Sin(angle)
		a := float64(x[left])
		b := float64(x[right])
		out[left] = float32(a*cos - b*sin)
		out[right] = float32(a*sin + b*cos)
	}
	return out, nil
}

func validateVectorPair(name string, a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("%s length mismatch: %d != %d", name, len(a), len(b))
	}
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
			return fmt.Errorf("%s input contains non-finite value", name)
		}
	}
	return nil
}
