package nativehf

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

type TensorValues struct {
	Values []float32
	Tensor modelinfo.SafetensorsTensor
}

type CoreWeights struct {
	TokenEmbedding TensorValues
	OutputNorm     TensorValues
	Output         TensorValues
	TiedOutput     bool
}

type AttentionWeights struct {
	Q   TensorValues
	K   TensorValues
	V   TensorValues
	Out TensorValues
}

type MLPWeights struct {
	Gate TensorValues
	Up   TensorValues
	Down TensorValues
}

type BlockWeights struct {
	InputNorm TensorValues
	PostNorm  TensorValues
	Attention AttentionWeights
	MLP       MLPWeights
}

func (m *Model) LoadCoreWeights() (CoreWeights, error) {
	if m.info == nil || m.info.HFWeights == nil {
		return CoreWeights{}, fmt.Errorf("native-hf weight manifest is not loaded")
	}
	names := m.info.HFWeights
	embedding, err := m.loadTensorValues(names.TokenEmbedding)
	if err != nil {
		return CoreWeights{}, err
	}
	norm, err := m.loadTensorValues(names.OutputNorm)
	if err != nil {
		return CoreWeights{}, err
	}
	output := embedding
	if !names.TiedOutput {
		output, err = m.loadTensorValues(names.Output)
		if err != nil {
			return CoreWeights{}, err
		}
	}
	return CoreWeights{
		TokenEmbedding: embedding,
		OutputNorm:     norm,
		Output:         output,
		TiedOutput:     names.TiedOutput,
	}, nil
}

func (m *Model) LoadBlockWeights(block int) (BlockWeights, error) {
	if m.info == nil || m.info.HFWeights == nil {
		return BlockWeights{}, fmt.Errorf("native-hf weight manifest is not loaded")
	}
	if block < 0 || block >= len(m.info.HFWeights.Blocks) {
		return BlockWeights{}, fmt.Errorf("native-hf block %d is out of range", block)
	}
	names := m.info.HFWeights.Blocks[block]
	inputNorm, err := m.loadTensorValues(names.InputNorm)
	if err != nil {
		return BlockWeights{}, err
	}
	postNorm, err := m.loadTensorValues(names.PostNorm)
	if err != nil {
		return BlockWeights{}, err
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
		InputNorm: inputNorm,
		PostNorm:  postNorm,
		Attention: attention,
		MLP:       mlp,
	}, nil
}

func (m *Model) loadAttentionWeights(names modelinfo.HFBlockTensorNames) (AttentionWeights, error) {
	q, err := m.loadTensorValues(names.AttentionQ)
	if err != nil {
		return AttentionWeights{}, err
	}
	k, err := m.loadTensorValues(names.AttentionK)
	if err != nil {
		return AttentionWeights{}, err
	}
	v, err := m.loadTensorValues(names.AttentionV)
	if err != nil {
		return AttentionWeights{}, err
	}
	out, err := m.loadTensorValues(names.AttentionOut)
	if err != nil {
		return AttentionWeights{}, err
	}
	return AttentionWeights{Q: q, K: k, V: v, Out: out}, nil
}

func (m *Model) loadMLPWeights(names modelinfo.HFBlockTensorNames) (MLPWeights, error) {
	gate, err := m.loadTensorValues(names.FFNGate)
	if err != nil {
		return MLPWeights{}, err
	}
	up, err := m.loadTensorValues(names.FFNUp)
	if err != nil {
		return MLPWeights{}, err
	}
	down, err := m.loadTensorValues(names.FFNDown)
	if err != nil {
		return MLPWeights{}, err
	}
	return MLPWeights{Gate: gate, Up: up, Down: down}, nil
}

func (m *Model) loadTensorValues(name string) (TensorValues, error) {
	if name == "" {
		return TensorValues{}, fmt.Errorf("native-hf tensor name is empty")
	}
	values, tensor, err := m.LoadTensorFloat32(name)
	if err != nil {
		return TensorValues{}, fmt.Errorf("load %s: %w", name, err)
	}
	return TensorValues{Values: values, Tensor: tensor}, nil
}
