package native

import (
	"fmt"
	"strings"
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
