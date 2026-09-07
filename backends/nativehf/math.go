package nativehf

import (
	"fmt"
	"math"

	"github.com/gakon/nego-ai/modelinfo"
)

func (m *Model) EmbedToken(tokenID int) ([]float32, error) {
	if m.spec == nil {
		return nil, fmt.Errorf("native-hf model spec is not loaded")
	}
	if m.info == nil || m.info.HFWeights == nil {
		return nil, fmt.Errorf("native-hf weight manifest is not loaded")
	}
	embedding, err := m.loadTensorValues(m.info.HFWeights.TokenEmbedding)
	if err != nil {
		return nil, err
	}
	return embeddingLookupFloat32(embedding.Values, embedding.Tensor, tokenID, m.spec.EmbeddingLength)
}

func (m *Model) OutputLogits(hidden []float32) ([]float32, error) {
	if m.spec == nil {
		return nil, fmt.Errorf("native-hf model spec is not loaded")
	}
	if m.info == nil || m.info.HFWeights == nil {
		return nil, fmt.Errorf("native-hf weight manifest is not loaded")
	}
	norm, err := m.loadTensorValues(m.info.HFWeights.OutputNorm)
	if err != nil {
		return nil, err
	}
	outputName := m.info.HFWeights.Output
	if m.info.HFWeights.TiedOutput {
		outputName = m.info.HFWeights.TokenEmbedding
	}
	output, err := m.loadTensorValues(outputName)
	if err != nil {
		return nil, err
	}
	normalized, err := rmsNormFloat32(hidden, norm.Values, m.spec.RMSNormEpsilon)
	if err != nil {
		return nil, err
	}
	return logitsFromOutputWeightFloat32(normalized, output.Values, output.Tensor)
}

func embeddingLookupFloat32(values []float32, tensor modelinfo.SafetensorsTensor, tokenID int, embeddingLength uint64) ([]float32, error) {
	vocab, dim, err := hfMatrixShape(tensor)
	if err != nil {
		return nil, err
	}
	if dim != embeddingLength {
		return nil, fmt.Errorf("embedding tensor %q dim %d does not match expected dim %d", tensor.Name, dim, embeddingLength)
	}
	if tokenID < 0 || tokenID >= int(vocab) {
		return nil, fmt.Errorf("token id %d is out of embedding range 0..%d", tokenID, vocab-1)
	}
	if len(values) != int(vocab*dim) {
		return nil, fmt.Errorf("embedding tensor %q has %d values, want %d", tensor.Name, len(values), vocab*dim)
	}
	start := tokenID * int(dim)
	out := make([]float32, int(dim))
	copy(out, values[start:start+int(dim)])
	return out, nil
}

func logitsFromOutputWeightFloat32(hidden, values []float32, tensor modelinfo.SafetensorsTensor) ([]float32, error) {
	if len(hidden) == 0 {
		return nil, fmt.Errorf("hidden state is empty")
	}
	vocab, dim, err := hfMatrixShape(tensor)
	if err != nil {
		return nil, err
	}
	if uint64(len(hidden)) != dim {
		return nil, fmt.Errorf("hidden state length %d does not match output dim %d", len(hidden), dim)
	}
	if len(values) != int(vocab*dim) {
		return nil, fmt.Errorf("output tensor %q has %d values, want %d", tensor.Name, len(values), vocab*dim)
	}
	logits := make([]float32, int(vocab))
	for token := 0; token < int(vocab); token++ {
		start := token * int(dim)
		sum, err := dotFloat32(hidden, values[start:start+int(dim)])
		if err != nil {
			return nil, err
		}
		logits[token] = sum
	}
	return logits, nil
}

func linearFloat32(input, values []float32, tensor modelinfo.SafetensorsTensor) ([]float32, error) {
	rows, cols, err := hfMatrixShape(tensor)
	if err != nil {
		return nil, err
	}
	if len(input) != int(cols) {
		return nil, fmt.Errorf("linear input length %d does not match tensor %q input dim %d", len(input), tensor.Name, cols)
	}
	if len(values) != int(rows*cols) {
		return nil, fmt.Errorf("linear tensor %q has %d values, want %d", tensor.Name, len(values), rows*cols)
	}
	out := make([]float32, int(rows))
	for row := 0; row < int(rows); row++ {
		start := row * int(cols)
		sum, err := dotFloat32(values[start:start+int(cols)], input)
		if err != nil {
			return nil, err
		}
		out[row] = sum
	}
	return out, nil
}

func mlpFloat32(input []float32, weights MLPWeights) ([]float32, error) {
	gate, err := linearFloat32(input, weights.Gate.Values, weights.Gate.Tensor)
	if err != nil {
		return nil, fmt.Errorf("gate projection: %w", err)
	}
	up, err := linearFloat32(input, weights.Up.Values, weights.Up.Tensor)
	if err != nil {
		return nil, fmt.Errorf("up projection: %w", err)
	}
	activated, err := siluFloat32(gate)
	if err != nil {
		return nil, err
	}
	hidden, err := mulFloat32(activated, up)
	if err != nil {
		return nil, err
	}
	return linearFloat32(hidden, weights.Down.Values, weights.Down.Tensor)
}

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

func mulFloat32(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("mul length mismatch: %d != %d", len(a), len(b))
	}
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out, nil
}

func dotFloat32(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("dot length mismatch: %d != %d", len(a), len(b))
	}
	var sum float32
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
			return 0, fmt.Errorf("dot input contains non-finite value")
		}
		sum += a[i] * b[i]
	}
	return sum, nil
}

func hfMatrixShape(tensor modelinfo.SafetensorsTensor) (uint64, uint64, error) {
	if len(tensor.Shape) != 2 {
		return 0, 0, fmt.Errorf("tensor %q must be 2D, got shape %v", tensor.Name, tensor.Shape)
	}
	if tensor.Shape[0] == 0 || tensor.Shape[1] == 0 {
		return 0, 0, fmt.Errorf("tensor %q has an empty matrix dimension", tensor.Name)
	}
	if tensor.Shape[0] > uint64(int(^uint(0)>>1)) || tensor.Shape[1] > uint64(int(^uint(0)>>1)) {
		return 0, 0, fmt.Errorf("tensor %q shape overflows this runtime", tensor.Name)
	}
	if tensor.Shape[0] > uint64(int(^uint(0)>>1))/tensor.Shape[1] {
		return 0, 0, fmt.Errorf("tensor %q value count overflows this runtime", tensor.Name)
	}
	return tensor.Shape[0], tensor.Shape[1], nil
}
