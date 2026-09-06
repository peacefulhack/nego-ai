package native

import (
	"math"
	"testing"
)

func TestSamplerGreedy(t *testing.T) {
	sampler := NewSampler(SamplingOptions{Temperature: 0})
	got, err := sampler.Sample([]float32{0.1, 3, 2})
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("got %d, want 1", got)
	}
}

func TestSamplerTopK(t *testing.T) {
	sampler := NewSampler(SamplingOptions{Temperature: 1, TopK: 1, Seed: 42})
	got, err := sampler.Sample([]float32{0.1, 3, 2})
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("got %d, want 1", got)
	}
}

func TestSamplerTopP(t *testing.T) {
	sampler := NewSampler(SamplingOptions{Temperature: 1, TopP: 0.5, Seed: 42})
	got, err := sampler.Sample([]float32{10, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("got %d, want 0", got)
	}
}

func TestSamplerRepeatPenalty(t *testing.T) {
	sampler := NewSampler(SamplingOptions{Temperature: 0, RepeatPenalty: 2})
	got, err := sampler.SampleWithHistory([]float32{10, 9}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("got %d, want 1", got)
	}
}

func TestSamplerRejectsInvalidLogits(t *testing.T) {
	sampler := NewSampler(SamplingOptions{Temperature: 1})
	if _, err := sampler.Sample(nil); err == nil {
		t.Fatal("expected empty logits error")
	}
	if _, err := sampler.Sample([]float32{0, float32(math.NaN())}); err == nil {
		t.Fatal("expected NaN logits error")
	}
	if _, err := sampler.Sample([]float32{0, float32(math.Inf(1))}); err == nil {
		t.Fatal("expected Inf logits error")
	}
}

func TestSamplerRejectsInvalidOptions(t *testing.T) {
	if _, err := (*Sampler)(nil).Sample([]float32{1}); err == nil {
		t.Fatal("expected nil sampler error")
	}
	if _, err := NewSampler(SamplingOptions{Temperature: float32(math.NaN())}).Sample([]float32{1}); err == nil {
		t.Fatal("expected NaN temperature error")
	}
	if _, err := NewSampler(SamplingOptions{Temperature: 1, TopP: float32(math.Inf(1))}).Sample([]float32{1}); err == nil {
		t.Fatal("expected Inf top-p error")
	}
	if _, err := NewSampler(SamplingOptions{RepeatPenalty: 0.5}).SampleWithHistory([]float32{1}, []int{0}); err == nil {
		t.Fatal("expected invalid repeat penalty error")
	}
}
