package native

import (
	"fmt"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

type GenerationOptions struct {
	MaxTokens int
	Sampling  SamplingOptions
	Stop      []string
}

func generationOptions(req nego.GenerateRequest) GenerationOptions {
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 16
	}
	return GenerationOptions{
		MaxTokens: maxTokens,
		Sampling: SamplingOptions{
			Temperature: float32(req.Temperature),
			TopP:        float32(req.TopP),
			Seed:        req.Seed,
		},
		Stop: append([]string(nil), req.Stop...),
	}
}

func chatGenerationOptions(req nego.ChatRequest) GenerationOptions {
	return generationOptions(nego.GenerateRequest{
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
		Seed:        req.Seed,
	})
}

func sampleTokenText(logits []float32, vocab *modelinfo.GGUFVocab, options SamplingOptions) (int, string, error) {
	return sampleTokenTextWithSampler(logits, vocab, NewSampler(options))
}

func sampleTokenTextWithSampler(logits []float32, vocab *modelinfo.GGUFVocab, sampler *Sampler) (int, string, error) {
	if vocab == nil {
		return 0, "", fmt.Errorf("gguf vocab is nil")
	}
	id, err := sampler.Sample(logits)
	if err != nil {
		return 0, "", err
	}
	text, err := vocab.Decode([]int{id}, modelinfo.DecodeOptions{})
	if err != nil {
		return 0, "", err
	}
	return id, text, nil
}
