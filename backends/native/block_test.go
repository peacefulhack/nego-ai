package native

import (
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestTransformerBlockFloat32(t *testing.T) {
	spec := ModelSpec{
		EmbeddingLength:    2,
		FeedForwardLength:  2,
		AttentionHeadCount: 1,
		KVHeadCount:        1,
		RopeTheta:          10000,
		RMSNormEpsilon:     0,
	}
	identity := []float32{
		1, 0,
		0, 1,
	}
	tensor := modelinfo.GGUFTensor{Name: "weight", Shape: []uint64{2, 2}}
	out, err := transformerBlockFloat32([]float32{1, 0}, BlockWeights{
		AttentionNorm: []float32{1, 1},
		FFNNorm:       []float32{1, 1},
		Attention: AttentionWeights{
			QValues:   identity,
			KValues:   identity,
			VValues:   identity,
			OutValues: identity,
			QTensor:   tensor,
			KTensor:   tensor,
			VTensor:   tensor,
			OutTensor: tensor,
		},
		MLP: MLPWeights{
			GateValues: identity,
			UpValues:   identity,
			DownValues: identity,
			GateTensor: tensor,
			UpTensor:   tensor,
			DownTensor: tensor,
		},
	}, spec, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0] <= 2 {
		t.Fatalf("unexpected block output: %#v", out)
	}
}
