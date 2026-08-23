package native

import (
	"context"
	"fmt"
	"strings"

	nego "github.com/gakon/nego-ai"
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
	return m.forwardToken(tokenID, position, nil)
}

func (m *Model) ForwardTokenWithState(tokenID int, state *DecodeState) ([]float32, error) {
	if state == nil {
		return nil, fmt.Errorf("decode state is nil")
	}
	logits, err := m.forwardToken(tokenID, state.Position, state.Cache)
	if err != nil {
		return nil, err
	}
	if err := state.advance(); err != nil {
		return nil, err
	}
	return logits, nil
}

func (m *Model) forwardToken(tokenID int, position int, cache *KVCache) ([]float32, error) {
	if !m.manifest.Ready() {
		return nil, fmt.Errorf("native tensor manifest is not ready: missing=%d shape_errors=%d", len(m.manifest.Missing), len(m.manifest.MissingShape))
	}
	embeddingValues, embeddingTensor, err := m.loadTensorFloat32Shared(m.names.TokenEmbedding)
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
		hidden, err = transformerBlockWithStateFloat32(hidden, weights, m.spec, i, position, cache)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
	}
	normValues, _, err := m.loadTensorFloat32Shared(m.names.OutputNorm)
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

func (m *Model) generateText(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	return m.generateTextWithEmitter(ctx, req, nil)
}

func (m *Model) generateTextWithEmitter(ctx context.Context, req nego.GenerateRequest, emit func(string) error) (*nego.GenerateOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	options := generationOptions(req)
	plan, err := m.planGeneration(req.Prompt, options)
	if err != nil {
		return nil, err
	}
	if len(plan.PromptTokenIDs) == 0 {
		return nil, m.inferenceError()
	}
	if !m.manifest.Ready() {
		return nil, m.inferenceError()
	}
	state, err := NewDecodeState(m.spec)
	if err != nil {
		return nil, err
	}
	var logits []float32
	for _, id := range plan.PromptTokenIDs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		logits, err = m.ForwardTokenWithState(id, state)
		if err != nil {
			return nil, err
		}
	}
	sampler := NewSampler(plan.Options.Sampling)
	history := append([]int(nil), plan.PromptTokenIDs...)
	var b strings.Builder
	emittedLen := 0
	for step := 0; step < plan.Options.MaxTokens; step++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		nextID, text, err := sampleTokenTextWithHistory(logits, m.vocab, sampler, history)
		if err != nil {
			return nil, err
		}
		history = append(history, nextID)
		if isEOSToken(m.vocab, nextID) {
			return &nego.GenerateOutput{Text: b.String()}, nil
		}
		b.WriteString(text)
		out := b.String()
		if stopReached(out, plan.Options.Stop) {
			trimmed := trimAtStop(out, plan.Options.Stop)
			if err := emitDelta(emit, trimmed, &emittedLen); err != nil {
				return nil, err
			}
			return &nego.GenerateOutput{Text: trimmed}, nil
		}
		if err := emitDelta(emit, out, &emittedLen); err != nil {
			return nil, err
		}
		if step == plan.Options.MaxTokens-1 {
			break
		}
		logits, err = m.ForwardTokenWithState(nextID, state)
		if err != nil {
			return nil, err
		}
	}
	return &nego.GenerateOutput{Text: b.String()}, nil
}

func emitDelta(emit func(string) error, text string, emittedLen *int) error {
	if emit == nil || emittedLen == nil || len(text) <= *emittedLen {
		return nil
	}
	delta := text[*emittedLen:]
	*emittedLen = len(text)
	if delta == "" {
		return nil
	}
	return emit(delta)
}

func stopReached(text string, stops []string) bool {
	for _, stop := range stops {
		if stop != "" && strings.Contains(text, stop) {
			return true
		}
	}
	return false
}

func trimAtStop(text string, stops []string) string {
	end := len(text)
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		if idx := strings.Index(text, stop); idx >= 0 && idx < end {
			end = idx
		}
	}
	return text[:end]
}

func (m *Model) outputWeights() ([]float32, modelinfo.GGUFTensor, error) {
	values, tensor, err := m.loadTensorFloat32Shared(m.names.Output)
	if err == nil {
		return values, tensor, nil
	}
	if !m.manifest.TiedOutput {
		return nil, modelinfo.GGUFTensor{}, err
	}
	values, tensor, tiedErr := m.loadTensorFloat32Shared(m.names.TokenEmbedding)
	if tiedErr != nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("load tied output weights: %w", tiedErr)
	}
	return values, tensor, nil
}
