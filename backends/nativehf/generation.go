package nativehf

import (
	"context"
	"fmt"
	"strings"

	nego "github.com/gakon/nego-ai"
)

func (m *Model) generateText(ctx context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m.tokenizer == nil {
		return nil, fmt.Errorf("native-hf tokenizer is not loaded")
	}
	if strings.TrimSpace(req.Prompt) == "" {
		return nil, fmt.Errorf("native-hf generation prompt is empty")
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 16
	}
	ids, err := m.tokenizer.Encode(req.Prompt)
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("native-hf generation prompt encoded to no tokens")
	}
	if m.spec == nil {
		return nil, fmt.Errorf("native-hf model spec is not loaded")
	}
	state, err := NewDecodeState(*m.spec)
	if err != nil {
		return nil, err
	}
	sampler := NewSampler(SamplingOptions{
		Temperature:   float32(req.Temperature),
		TopP:          float32(req.TopP),
		RepeatPenalty: float32(req.RepeatPenalty),
		Seed:          req.Seed,
	})
	history := append([]int(nil), ids...)
	var logits []float32
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		logits, err = m.ForwardTokenWithState(id, state)
		if err != nil {
			return nil, err
		}
	}
	var b strings.Builder
	for i := 0; i < maxTokens; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		nextID, err := sampler.SampleWithHistory(m.applyAdapter(logits), history)
		if err != nil {
			return nil, err
		}
		history = append(history, nextID)
		text, err := m.tokenizer.Decode([]int{nextID})
		if err != nil {
			return nil, err
		}
		b.WriteString(text)
		out := b.String()
		if stopReached(out, req.Stop) {
			return &nego.GenerateOutput{Text: trimAtStop(out, req.Stop)}, nil
		}
		if i == maxTokens-1 {
			break
		}
		logits, err = m.ForwardTokenWithState(nextID, state)
		if err != nil {
			return nil, err
		}
	}
	return &nego.GenerateOutput{Text: b.String()}, nil
}

func (m *Model) applyAdapter(logits []float32) []float32 {
	if m.adapter == nil {
		return logits
	}
	return m.adapter.Apply(logits)
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
