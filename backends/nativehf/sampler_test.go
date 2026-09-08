package nativehf

import (
	"math"
	"testing"
)

func TestSamplerRepeatPenaltyCanChangeGreedyToken(t *testing.T) {
	sampler := NewSampler(SamplingOptions{RepeatPenalty: 2})
	id, err := sampler.SampleWithHistory([]float32{10, 9}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("sampled token = %d, want 1", id)
	}
}

func TestSamplerTopPIsDeterministicWithSeed(t *testing.T) {
	sampler := NewSampler(SamplingOptions{Temperature: 1, TopP: 0.5, Seed: 42})
	id, err := sampler.SampleWithHistory([]float32{10, 1, 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if id != 0 {
		t.Fatalf("sampled token = %d, want 0", id)
	}
}

func TestSamplerRejectsInvalidOptions(t *testing.T) {
	if _, err := NewSampler(SamplingOptions{Temperature: float32(math.NaN())}).SampleWithHistory([]float32{1}, nil); err == nil {
		t.Fatal("expected NaN temperature error")
	}
	if _, err := NewSampler(SamplingOptions{RepeatPenalty: 0.5}).SampleWithHistory([]float32{1}, []int{0}); err == nil {
		t.Fatal("expected repeat penalty error")
	}
}
