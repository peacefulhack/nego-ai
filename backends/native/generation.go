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
			Temperature:   float32(req.Temperature),
			TopK:          req.TopK,
			TopP:          float32(req.TopP),
			RepeatPenalty: float32(req.RepeatPenalty),
			Seed:          req.Seed,
		},
		Stop: append([]string(nil), req.Stop...),
	}
}

func chatGenerationOptions(req nego.ChatRequest) GenerationOptions {
	return generationOptions(nego.GenerateRequest{
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		RepeatPenalty: req.RepeatPenalty,
		Stop:          req.Stop,
		Seed:          req.Seed,
	})
}

func sampleTokenText(logits []float32, vocab *modelinfo.GGUFVocab, options SamplingOptions) (int, string, error) {
	return sampleTokenTextWithSampler(logits, vocab, NewSampler(options))
}

func sampleTokenTextWithSampler(logits []float32, vocab *modelinfo.GGUFVocab, sampler *Sampler) (int, string, error) {
	return sampleTokenTextWithHistory(logits, vocab, sampler, nil)
}

func sampleTokenTextWithHistory(logits []float32, vocab *modelinfo.GGUFVocab, sampler *Sampler, history []int) (int, string, error) {
	if vocab == nil {
		return 0, "", fmt.Errorf("gguf vocab is nil")
	}
	id, err := sampler.SampleWithHistory(logits, history)
	if err != nil {
		return 0, "", err
	}
	text, err := vocab.Decode([]int{id}, modelinfo.DecodeOptions{})
	if err != nil {
		return 0, "", err
	}
	return id, text, nil
}

func isEOSToken(vocab *modelinfo.GGUFVocab, id int) bool {
	if vocab == nil || id < 0 || id >= len(vocab.Tokens) {
		return false
	}
	if vocab.EOSTokenID != 0 && id == int(vocab.EOSTokenID) {
		return true
	}
	switch vocab.Tokens[id] {
	case "</s>", "<eos>", "<|endoftext|>", "<|end_of_text|>", "<|im_end|>":
		return true
	default:
		return false
	}
}
