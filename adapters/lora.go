package adapters

import (
	"fmt"
	"math"
	"math/rand"
)

const maxLoRAParameters = 8 << 20

// LinearLoRA adds (Alpha/Rank)*B*A*x to a frozen linear projection. A is
// row-major [Rank, InputSize]; B is row-major [OutputSize, Rank]. Callers must
// not mutate weights concurrently with Forward, Backward, or StepSGD.
// Reference: https://arxiv.org/abs/2106.09685
type LinearLoRA struct {
	Version    int       `json:"version"`
	Type       string    `json:"type"`
	InputSize  int       `json:"input_size"`
	OutputSize int       `json:"output_size"`
	Rank       int       `json:"rank"`
	Alpha      float32   `json:"alpha"`
	A          []float32 `json:"a"`
	B          []float32 `json:"b"`
}

// NewLinearLoRA initializes A from a seeded normal distribution and B to zero,
// so attaching a new adapter leaves the frozen projection unchanged.
func NewLinearLoRA(inputSize, outputSize, rank int, alpha float32, seed int64) (*LinearLoRA, error) {
	l := &LinearLoRA{Version: CurrentVersion, Type: "lora_linear", InputSize: inputSize, OutputSize: outputSize, Rank: rank, Alpha: alpha}
	if err := l.validateShape(); err != nil {
		return nil, err
	}
	l.A, l.B = make([]float32, rank*inputSize), make([]float32, outputSize*rank)
	rng := rand.New(rand.NewSource(seed))
	stddev := 1 / math.Sqrt(float64(inputSize))
	for i := range l.A {
		l.A[i] = float32(rng.NormFloat64() * stddev)
	}
	return l, nil
}

func (l *LinearLoRA) validateShape() error {
	if l == nil {
		return fmt.Errorf("linear LoRA is nil")
	}
	if l.Version != CurrentVersion || l.Type != "lora_linear" {
		return fmt.Errorf("unsupported linear LoRA format")
	}
	if l.InputSize <= 0 || l.OutputSize <= 0 || l.Rank <= 0 || l.Rank > l.InputSize || l.Rank > l.OutputSize {
		return fmt.Errorf("invalid linear LoRA dimensions")
	}
	// Bound each product before multiplication, including on 32-bit hosts.
	if l.InputSize > maxLoRAParameters/l.Rank || l.OutputSize > maxLoRAParameters/l.Rank ||
		l.InputSize*l.Rank > maxLoRAParameters-l.OutputSize*l.Rank {
		return fmt.Errorf("linear LoRA exceeds %d parameters", maxLoRAParameters)
	}
	if !finiteLoRA(l.Alpha) || l.Alpha <= 0 {
		return fmt.Errorf("linear LoRA alpha must be finite and positive")
	}
	if l.Alpha/float32(l.Rank) == 0 {
		return fmt.Errorf("linear LoRA scale underflows float32")
	}
	return nil
}

func (l *LinearLoRA) Validate() error {
	if err := l.validateShape(); err != nil {
		return err
	}
	if len(l.A) != l.Rank*l.InputSize || len(l.B) != l.OutputSize*l.Rank {
		return fmt.Errorf("linear LoRA weight lengths do not match dimensions")
	}
	if err := checkLoRAVector("A", l.A); err != nil {
		return err
	}
	return checkLoRAVector("B", l.B)
}

// Forward returns frozenOutput + (alpha/rank)*B*A*input without changing inputs.
// frozenOutput must be computed using the unchanged base model weights.
func (l *LinearLoRA) Forward(input, frozenOutput []float32) ([]float32, error) {
	if err := l.Validate(); err != nil {
		return nil, err
	}
	if len(input) != l.InputSize || len(frozenOutput) != l.OutputSize {
		return nil, fmt.Errorf("linear LoRA forward shape mismatch")
	}
	if err := checkLoRAVector("input", input); err != nil {
		return nil, err
	}
	if err := checkLoRAVector("frozen output", frozenOutput); err != nil {
		return nil, err
	}
	z := l.projectA(input)
	out := make([]float32, l.OutputSize)
	scale := float64(l.Alpha) / float64(l.Rank)
	for row := range out {
		var delta float64
		for k := 0; k < l.Rank; k++ {
			delta += float64(l.B[row*l.Rank+k]) * z[k]
		}
		out[row] = float32(float64(frozenOutput[row]) + scale*delta)
	}
	if err := checkLoRAVector("output", out); err != nil {
		return nil, err
	}
	return out, nil
}

// LinearLoRAGradients contains parameter gradients and the adapter branch's
// input gradient. Input does NOT include the frozen base projection's gradient.
type LinearLoRAGradients struct {
	A, B, Input []float32
}

// Backward computes vector-Jacobian products from an upstream output gradient.
// Use the same weights and input as Forward; it does not mutate the adapter.
func (l *LinearLoRA) Backward(input, outputGradient []float32) (LinearLoRAGradients, error) {
	if err := l.Validate(); err != nil {
		return LinearLoRAGradients{}, err
	}
	if len(input) != l.InputSize || len(outputGradient) != l.OutputSize {
		return LinearLoRAGradients{}, fmt.Errorf("linear LoRA backward shape mismatch")
	}
	if err := checkLoRAVector("input", input); err != nil {
		return LinearLoRAGradients{}, err
	}
	if err := checkLoRAVector("output gradient", outputGradient); err != nil {
		return LinearLoRAGradients{}, err
	}
	g := LinearLoRAGradients{A: make([]float32, len(l.A)), B: make([]float32, len(l.B)), Input: make([]float32, l.InputSize)}
	z, dz := l.projectA(input), make([]float64, l.Rank)
	scale := float64(l.Alpha) / float64(l.Rank)
	for row, dy := range outputGradient {
		for k := 0; k < l.Rank; k++ {
			g.B[row*l.Rank+k] = float32(scale * float64(dy) * z[k])
			dz[k] += scale * float64(dy) * float64(l.B[row*l.Rank+k])
		}
	}
	for col, x := range input {
		var dx float64
		for k := 0; k < l.Rank; k++ {
			g.A[k*l.InputSize+col] = float32(dz[k] * float64(x))
			dx += dz[k] * float64(l.A[k*l.InputSize+col])
		}
		g.Input[col] = float32(dx)
	}
	for name, values := range map[string][]float32{"A gradient": g.A, "B gradient": g.B, "input gradient": g.Input} {
		if err := checkLoRAVector(name, values); err != nil {
			return LinearLoRAGradients{}, err
		}
	}
	return g, nil
}

// StepSGD applies one unregularized SGD update. It validates both proposed
// matrices before committing either, so invalid gradients cannot partly update
// the adapter. Callers handle batching, clipping, and loss reduction explicitly.
func (l *LinearLoRA) StepSGD(g LinearLoRAGradients, learningRate float32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if !finiteLoRA(learningRate) || learningRate <= 0 {
		return fmt.Errorf("LoRA learning rate must be finite and positive")
	}
	if len(g.A) != len(l.A) || len(g.B) != len(l.B) {
		return fmt.Errorf("LoRA gradient shape mismatch")
	}
	if err := checkLoRAVector("A gradient", g.A); err != nil {
		return err
	}
	if err := checkLoRAVector("B gradient", g.B); err != nil {
		return err
	}
	a, b := make([]float32, len(l.A)), make([]float32, len(l.B))
	for i := range a {
		a[i] = float32(float64(l.A[i]) - float64(learningRate)*float64(g.A[i]))
	}
	for i := range b {
		b[i] = float32(float64(l.B[i]) - float64(learningRate)*float64(g.B[i]))
	}
	if err := checkLoRAVector("updated A", a); err != nil {
		return err
	}
	if err := checkLoRAVector("updated B", b); err != nil {
		return err
	}
	l.A, l.B = a, b
	return nil
}

func (l *LinearLoRA) projectA(input []float32) []float64 {
	z := make([]float64, l.Rank)
	for k := range z {
		for col, x := range input {
			z[k] += float64(l.A[k*l.InputSize+col]) * float64(x)
		}
	}
	return z
}

func finiteLoRA(value float32) bool { return math.Float32bits(value)&0x7f800000 != 0x7f800000 }

func checkLoRAVector(name string, values []float32) error {
	for i, value := range values {
		if !finiteLoRA(value) {
			return fmt.Errorf("LoRA %s contains a non-finite value at %d", name, i)
		}
	}
	return nil
}
