package native

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

type AttentionWeights struct {
	QValues   []float32
	KValues   []float32
	VValues   []float32
	OutValues []float32
	QTensor   modelinfo.GGUFTensor
	KTensor   modelinfo.GGUFTensor
	VTensor   modelinfo.GGUFTensor
	OutTensor modelinfo.GGUFTensor
}

type MLPWeights struct {
	GateValues []float32
	UpValues   []float32
	DownValues []float32
	GateTensor modelinfo.GGUFTensor
	UpTensor   modelinfo.GGUFTensor
	DownTensor modelinfo.GGUFTensor
}

type BlockWeights struct {
	AttentionNorm []float32
	FFNNorm       []float32
	Attention     AttentionWeights
	MLP           MLPWeights
}

func transformerBlockFloat32(input []float32, weights BlockWeights, spec ModelSpec, position int) ([]float32, error) {
	return transformerBlockWithStateFloat32(input, weights, spec, -1, position, nil)
}

func transformerBlockWithStateFloat32(input []float32, weights BlockWeights, spec ModelSpec, layer int, position int, cache *KVCache) ([]float32, error) {
	attnInput, err := rmsNormFloat32(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	if err != nil {
		return nil, fmt.Errorf("attention rmsnorm: %w", err)
	}
	attnOut, err := multiHeadAttentionWithCacheFloat32(attnInput, weights.Attention.QValues, weights.Attention.KValues, weights.Attention.VValues, weights.Attention.OutValues, weights.Attention.QTensor, weights.Attention.KTensor, weights.Attention.VTensor, weights.Attention.OutTensor, spec, layer, position, cache)
	if err != nil {
		return nil, fmt.Errorf("attention: %w", err)
	}
	residual, err := addFloat32(input, attnOut)
	if err != nil {
		return nil, fmt.Errorf("attention residual: %w", err)
	}
	mlpInput, err := rmsNormFloat32(residual, weights.FFNNorm, spec.RMSNormEpsilon)
	if err != nil {
		return nil, fmt.Errorf("ffn rmsnorm: %w", err)
	}
	mlpOut, err := mlpFloat32(mlpInput, weights.MLP.GateValues, weights.MLP.UpValues, weights.MLP.DownValues, weights.MLP.GateTensor, weights.MLP.UpTensor, weights.MLP.DownTensor)
	if err != nil {
		return nil, fmt.Errorf("ffn: %w", err)
	}
	out, err := addFloat32(residual, mlpOut)
	if err != nil {
		return nil, fmt.Errorf("ffn residual: %w", err)
	}
	return out, nil
}
