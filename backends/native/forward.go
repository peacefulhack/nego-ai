package native

import (
	"fmt"
	"strings"

	"github.com/gakon/nego-ai/modelinfo"
)

type GenerationPlan struct {
	PromptTokenIDs []int
	Options        GenerationOptions
}

func (m *Model) planGeneration(prompt string, options GenerationOptions) (GenerationPlan, error) {
	if strings.TrimSpace(prompt) == "" {
		return GenerationPlan{}, fmt.Errorf("native generation prompt is empty")
	}
	plan := GenerationPlan{Options: options}
	if m.vocab == nil || len(m.vocab.Tokens) == 0 {
		return plan, nil
	}
	ids, err := m.EncodeText(prompt)
	if err != nil {
		return GenerationPlan{}, err
	}
	if len(ids) == 0 {
		return GenerationPlan{}, fmt.Errorf("native generation prompt encoded to no tokens")
	}
	plan.PromptTokenIDs = ids
	return plan, nil
}

func (m *Model) ForwardToken(tokenID int, position int) ([]float32, error) {
	if !m.manifest.Ready() {
		return nil, fmt.Errorf("native tensor manifest is not ready: missing=%d shape_errors=%d", len(m.manifest.Missing), len(m.manifest.MissingShape))
	}
	embeddingValues, embeddingTensor, err := m.LoadTensorFloat32(m.names.TokenEmbedding)
	if err != nil {
		return nil, err
	}
	hidden, err := embeddingLookupFloat32(embeddingValues, embeddingTensor, tokenID, m.spec.EmbeddingLength)
	if err != nil {
		return nil, err
	}
	for i := range m.names.Blocks {
		weights, err := m.LoadBlockWeights(i)
		if err != nil {
			return nil, fmt.Errorf("load block %d: %w", i, err)
		}
		hidden, err = transformerBlockFloat32(hidden, weights, m.spec, position)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
	}
	normValues, _, err := m.LoadTensorFloat32(m.names.OutputNorm)
	if err != nil {
		return nil, err
	}
	hidden, err = rmsNormFloat32(hidden, normValues, m.spec.RMSNormEpsilon)
	if err != nil {
		return nil, fmt.Errorf("output rmsnorm: %w", err)
	}
	outputValues, outputTensor, err := m.outputWeights()
	if err != nil {
		return nil, err
	}
	return logitsFromOutputWeightFloat32(hidden, outputValues, outputTensor)
}

func (m *Model) outputWeights() ([]float32, modelinfo.GGUFTensor, error) {
	values, tensor, err := m.LoadTensorFloat32(m.names.Output)
	if err == nil {
		return values, tensor, nil
	}
	if !m.manifest.TiedOutput {
		return nil, modelinfo.GGUFTensor{}, err
	}
	values, tensor, tiedErr := m.LoadTensorFloat32(m.names.TokenEmbedding)
	if tiedErr != nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("load tied output weights: %w", tiedErr)
	}
	return values, tensor, nil
}
