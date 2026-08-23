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
	if len(x)%2 != 0 {
		return nil, fmt.Errorf("rope input length must be even")
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
		if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) || math.IsNaN(float64(x[i+1])) || math.IsInf(float64(x[i+1]), 0) {
			return nil, fmt.Errorf("rope input contains non-finite value")
		}
		freq := math.Pow(theta, -float64(i)/dim)
		angle := float64(position) * freq
		cos, sin := math.Cos(angle), math.Sin(angle)
		a := float64(x[i])
		b := float64(x[i+1])
		out[i] = float32(a*cos - b*sin)
		out[i+1] = float32(a*sin + b*cos)
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
