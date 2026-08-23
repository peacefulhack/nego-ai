package native

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

func (m *Model) LoadBlockWeights(block int) (BlockWeights, error) {
	if block < 0 || block >= len(m.names.Blocks) {
		return BlockWeights{}, fmt.Errorf("block %d is out of range", block)
	}
	names := m.names.Blocks[block]
	attentionNorm, _, err := m.LoadTensorFloat32(names.AttentionNorm)
	if err != nil {
		return BlockWeights{}, fmt.Errorf("%s: %w", names.AttentionNorm, err)
	}
	ffnNorm, _, err := m.LoadTensorFloat32(names.FFNNorm)
	if err != nil {
		return BlockWeights{}, fmt.Errorf("%s: %w", names.FFNNorm, err)
	}
	attention, err := m.loadAttentionWeights(names)
	if err != nil {
		return BlockWeights{}, err
	}
	mlp, err := m.loadMLPWeights(names)
	if err != nil {
		return BlockWeights{}, err
	}
	return BlockWeights{
		AttentionNorm: attentionNorm,
		FFNNorm:       ffnNorm,
		Attention:     attention,
		MLP:           mlp,
	}, nil
}

func (m *Model) loadAttentionWeights(names BlockTensorNames) (AttentionWeights, error) {
	qValues, qTensor, err := m.loadNamedFloat32(names.AttentionQ)
	if err != nil {
		return AttentionWeights{}, err
	}
	kValues, kTensor, err := m.loadNamedFloat32(names.AttentionK)
	if err != nil {
		return AttentionWeights{}, err
	}
	vValues, vTensor, err := m.loadNamedFloat32(names.AttentionV)
	if err != nil {
		return AttentionWeights{}, err
	}
	outValues, outTensor, err := m.loadNamedFloat32(names.AttentionOut)
	if err != nil {
		return AttentionWeights{}, err
	}
	return AttentionWeights{
		QValues:   qValues,
		KValues:   kValues,
		VValues:   vValues,
		OutValues: outValues,
		QTensor:   qTensor,
		KTensor:   kTensor,
		VTensor:   vTensor,
		OutTensor: outTensor,
	}, nil
}

func (m *Model) loadMLPWeights(names BlockTensorNames) (MLPWeights, error) {
	gateValues, gateTensor, err := m.loadNamedFloat32(names.FFNGate)
	if err != nil {
		return MLPWeights{}, err
	}
	upValues, upTensor, err := m.loadNamedFloat32(names.FFNUp)
	if err != nil {
		return MLPWeights{}, err
	}
	downValues, downTensor, err := m.loadNamedFloat32(names.FFNDown)
	if err != nil {
		return MLPWeights{}, err
	}
	return MLPWeights{
		GateValues: gateValues,
		UpValues:   upValues,
		DownValues: downValues,
		GateTensor: gateTensor,
		UpTensor:   upTensor,
		DownTensor: downTensor,
	}, nil
}

func (m *Model) loadNamedFloat32(name string) ([]float32, modelinfo.GGUFTensor, error) {
	values, tensor, err := m.LoadTensorFloat32(name)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("load %s: %w", name, err)
	}
	return values, tensor, nil
}
