package native

import (
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

func TestGenerationOptionsDefaults(t *testing.T) {
	opts := generationOptions(nego.GenerateRequest{})
	if opts.MaxTokens != 16 {
		t.Fatalf("MaxTokens = %d, want 16", opts.MaxTokens)
	}
	if opts.Sampling.Seed != 0 || opts.Sampling.Temperature != 0 {
		t.Fatalf("unexpected sampling defaults: %#v", opts.Sampling)
	}
}

func TestChatGenerationOptions(t *testing.T) {
	opts := chatGenerationOptions(nego.ChatRequest{
		MaxTokens:     4,
		Temperature:   0.5,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		Stop:          []string{"</s>"},
		Seed:          7,
	})
	if opts.MaxTokens != 4 || opts.Sampling.Temperature != 0.5 || opts.Sampling.TopP != 0.9 || opts.Sampling.RepeatPenalty != 1.1 || opts.Sampling.Seed != 7 {
		t.Fatalf("unexpected options: %#v", opts)
	}
	if len(opts.Stop) != 1 || opts.Stop[0] != "</s>" {
		t.Fatalf("unexpected stop options: %#v", opts.Stop)
	}
}

func TestIsEOSToken(t *testing.T) {
	vocab := &modelinfo.GGUFVocab{Tokens: []string{"hello", "</s>", "<|im_end|>"}}
	if !isEOSToken(vocab, 1) || !isEOSToken(vocab, 2) {
		t.Fatal("expected EOS token detection")
	}
	if isEOSToken(vocab, 0) {
		t.Fatal("unexpected EOS token detection")
	}
}

func TestSampleTokenText(t *testing.T) {
	vocab := &modelinfo.GGUFVocab{Tokens: []string{"▁hello", "▁world"}}
	id, text, err := sampleTokenText([]float32{0, 10}, vocab, SamplingOptions{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 || text != " world" {
		t.Fatalf("unexpected sampled token: %d %q", id, text)
	}
}

func TestSampleTokenTextRejectsNilVocab(t *testing.T) {
	_, _, err := sampleTokenText([]float32{1}, nil, SamplingOptions{})
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}
