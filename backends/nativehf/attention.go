package nativehf

import (
	"fmt"
	"math"

	"github.com/gakon/nego-ai/modelinfo"
)

func (m *Model) ForwardToken(tokenID int, position int) ([]float32, error) {
	if m.info == nil || m.info.HFWeights == nil {
		return nil, fmt.Errorf("native-hf weight manifest is not loaded")
	}
	if m.spec == nil {
		return nil, fmt.Errorf("native-hf model spec is not loaded")
	}
	if m.info.HFShapes != nil && !m.info.HFShapes.Ready {
		return nil, fmt.Errorf("native-hf tensor shapes are not ready: mismatches=%d", len(m.info.HFShapes.MissingShape))
	}
	hidden, err := m.EmbedToken(tokenID)
	if err != nil {
		return nil, err
	}
	for i := range m.info.HFWeights.Blocks {
		weights, err := m.LoadBlockWeights(i)
		if err != nil {
			return nil, fmt.Errorf("load block %d: %w", i, err)
		}
		hidden, err = transformerBlockFloat32(hidden, weights, *m.spec, position)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
	}
	return m.OutputLogits(hidden)
}

func transformerBlockFloat32(input []float32, weights BlockWeights, spec modelinfo.HFModelSpec, position int) ([]float32, error) {
	attnInput, err := rmsNormFloat32(input, weights.InputNorm.Values, spec.RMSNormEpsilon)
	if err != nil {
		return nil, fmt.Errorf("attention rmsnorm: %w", err)
	}
	attnOut, err := multiHeadAttentionFloat32(attnInput, weights.Attention, spec, position)
	if err != nil {
		return nil, fmt.Errorf("attention: %w", err)
	}
	residual, err := addFloat32(input, attnOut)
	if err != nil {
		return nil, fmt.Errorf("attention residual: %w", err)
	}
	mlpInput, err := rmsNormFloat32(residual, weights.PostNorm.Values, spec.RMSNormEpsilon)
	if err != nil {
		return nil, fmt.Errorf("ffn rmsnorm: %w", err)
	}
	mlpOut, err := mlpFloat32(mlpInput, weights.MLP)
	if err != nil {
		return nil, fmt.Errorf("ffn: %w", err)
	}
	out, err := addFloat32(residual, mlpOut)
	if err != nil {
		return nil, fmt.Errorf("ffn residual: %w", err)
	}
	return out, nil
}

func multiHeadAttentionFloat32(input []float32, weights AttentionWeights, spec modelinfo.HFModelSpec, position int) ([]float32, error) {
	if spec.EmbeddingLength > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("embedding length %d overflows this runtime", spec.EmbeddingLength)
	}
	if spec.AttentionHeadCount == 0 || spec.KVHeadCount == 0 || spec.AttentionHeadCount%spec.KVHeadCount != 0 {
		return nil, fmt.Errorf("invalid grouped-query dimensions: heads=%d kv_heads=%d", spec.AttentionHeadCount, spec.KVHeadCount)
	}
	if spec.HeadDim == 0 || spec.HeadDim > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("attention head dimension %d is invalid on this runtime", spec.HeadDim)
	}
	if len(input) != int(spec.EmbeddingLength) {
		return nil, fmt.Errorf("multi-head attention input length %d does not match embedding length %d", len(input), spec.EmbeddingLength)
	}
	headDim := int(spec.HeadDim)
	q, err := linearFloat32(input, weights.Q.Values, weights.Q.Tensor)
	if err != nil {
		return nil, fmt.Errorf("q projection: %w", err)
	}
	k, err := linearFloat32(input, weights.K.Values, weights.K.Tensor)
	if err != nil {
		return nil, fmt.Errorf("k projection: %w", err)
	}
	v, err := linearFloat32(input, weights.V.Values, weights.V.Tensor)
	if err != nil {
		return nil, fmt.Errorf("v projection: %w", err)
	}
	qHeads, err := splitHeads(q, int(spec.AttentionHeadCount), headDim)
	if err != nil {
		return nil, err
	}
	kHeads, err := splitHeads(k, int(spec.KVHeadCount), headDim)
	if err != nil {
		return nil, err
	}
	vHeads, err := splitHeads(v, int(spec.KVHeadCount), headDim)
	if err != nil {
		return nil, err
	}
	headsPerKV := int(spec.AttentionHeadCount / spec.KVHeadCount)
	concat := make([]float32, 0, len(q))
	for i, qHead := range qHeads {
		kvHead := i / headsPerKV
		if weights.QNorm != nil {
			qHead, err = rmsNormFloat32(qHead, weights.QNorm.Values, spec.RMSNormEpsilon)
			if err != nil {
				return nil, fmt.Errorf("q norm head %d: %w", i, err)
			}
		}
		kHead := kHeads[kvHead]
		if weights.KNorm != nil {
			kHead, err = rmsNormFloat32(kHead, weights.KNorm.Values, spec.RMSNormEpsilon)
			if err != nil {
				return nil, fmt.Errorf("k norm head %d: %w", kvHead, err)
			}
		}
		rotatedQ, err := applyRoPEFloat32(qHead, position, spec.RopeTheta)
		if err != nil {
			return nil, fmt.Errorf("q rope head %d: %w", i, err)
		}
		rotatedK, err := applyRoPEFloat32(kHead, position, spec.RopeTheta)
		if err != nil {
			return nil, fmt.Errorf("k rope head %d: %w", kvHead, err)
		}
		headOut, err := attentionFloat32(rotatedQ, [][]float32{rotatedK}, [][]float32{vHeads[kvHead]})
		if err != nil {
			return nil, fmt.Errorf("attention head %d: %w", i, err)
		}
		concat = append(concat, headOut...)
	}
	return linearFloat32(concat, weights.Out.Values, weights.Out.Tensor)
}

func splitHeads(values []float32, heads int, headDim int) ([][]float32, error) {
	if heads <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("heads and head dimension must be positive")
	}
	if heads > int(^uint(0)>>1)/headDim {
		return nil, fmt.Errorf("head shape overflows this runtime")
	}
	if len(values) != heads*headDim {
		return nil, fmt.Errorf("head split length mismatch: values=%d want=%d", len(values), heads*headDim)
	}
	out := make([][]float32, heads)
	for i := 0; i < heads; i++ {
		start := i * headDim
		out[i] = values[start : start+headDim]
	}
	return out, nil
}

func attentionFloat32(query []float32, keys, values [][]float32) ([]float32, error) {
	if len(query) == 0 {
		return nil, fmt.Errorf("attention query is empty")
	}
	if len(keys) == 0 || len(keys) != len(values) {
		return nil, fmt.Errorf("attention key/value length mismatch: keys=%d values=%d", len(keys), len(values))
	}
	scores := make([]float32, len(keys))
	scale := float32(1 / math.Sqrt(float64(len(query))))
	for i := range keys {
		if len(keys[i]) != len(query) || len(values[i]) != len(query) {
			return nil, fmt.Errorf("attention vector length mismatch at %d", i)
		}
		score, err := dotFloat32(query, keys[i])
		if err != nil {
			return nil, err
		}
		scores[i] = score * scale
	}
	probs, err := softmaxFloat32(scores)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(query))
	for i, prob := range probs {
		for j, value := range values[i] {
			out[j] += prob * value
		}
	}
	return out, nil
}

func softmaxFloat32(x []float32) ([]float32, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("softmax input is empty")
	}
	maxValue := x[0]
	for _, value := range x {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("softmax input contains non-finite value")
		}
		if value > maxValue {
			maxValue = value
		}
	}
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

func addFloat32(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("add length mismatch: %d != %d", len(a), len(b))
	}
	out := make([]float32, len(a))
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
			return nil, fmt.Errorf("add input contains non-finite value")
		}
		out[i] = a[i] + b[i]
	}
	return out, nil
}
