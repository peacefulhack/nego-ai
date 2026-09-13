package nativehf

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

type SamplingOptions struct {
	Temperature   float32
	TopK          int
	TopP          float32
	RepeatPenalty float32
	Seed          int64
}

type Sampler struct {
	options SamplingOptions
	rng     *rand.Rand
}

func NewSampler(options SamplingOptions) *Sampler {
	seed := options.Seed
	if seed == 0 {
		seed = 1
	}
	return &Sampler{
		options: options,
		rng:     rand.New(rand.NewSource(seed)),
	}
}

func (s *Sampler) SampleWithHistory(logits []float32, history []int) (int, error) {
	if s == nil {
		return 0, fmt.Errorf("native-hf sampler is nil")
	}
	if len(logits) == 0 {
		return 0, fmt.Errorf("logits are empty")
	}
	if err := validateLogits(logits); err != nil {
		return 0, err
	}
	adjusted, err := applyRepeatPenalty(logits, history, s.options.RepeatPenalty)
	if err != nil {
		return 0, err
	}
	temperature := float64(s.options.Temperature)
	if math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return 0, fmt.Errorf("temperature must be finite")
	}
	if temperature <= 0 {
		return argmax(adjusted), nil
	}
	topP := float64(s.options.TopP)
	if math.IsNaN(topP) || math.IsInf(topP, 0) {
		return 0, fmt.Errorf("top-p must be finite")
	}
	if s.options.TopK < 0 {
		return 0, fmt.Errorf("top-k must be >= 0")
	}
	candidates := sortedCandidates(adjusted)
	if s.options.TopK > 0 && s.options.TopK < len(candidates) {
		candidates = candidates[:s.options.TopK]
	}
	probs := softmaxCandidates(candidates, temperature)
	if topP > 0 && topP < 1 {
		candidates, probs = applyTopP(candidates, probs, topP)
	}
	return sampleCandidates(s.rng, candidates, probs), nil
}

type sampleCandidate struct {
	index int
	logit float32
}

func validateLogits(logits []float32) error {
	for i, logit := range logits {
		if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 0) {
			return fmt.Errorf("logit %d is not finite", i)
		}
	}
	return nil
}

func applyRepeatPenalty(logits []float32, history []int, penalty float32) ([]float32, error) {
	if penalty == 0 || penalty == 1 || len(history) == 0 {
		return logits, nil
	}
	value := float64(penalty)
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 1 {
		return nil, fmt.Errorf("repeat penalty must be finite and >= 1")
	}
	adjusted := append([]float32(nil), logits...)
	seen := make(map[int]struct{}, len(history))
	for _, id := range history {
		if id < 0 || id >= len(adjusted) {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		if adjusted[id] < 0 {
			adjusted[id] *= penalty
		} else {
			adjusted[id] /= penalty
		}
	}
	return adjusted, nil
}

func argmax(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

func sortedCandidates(logits []float32) []sampleCandidate {
	candidates := make([]sampleCandidate, len(logits))
	for i, logit := range logits {
		candidates[i] = sampleCandidate{index: i, logit: logit}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].logit > candidates[j].logit
	})
	return candidates
}

func softmaxCandidates(candidates []sampleCandidate, temperature float64) []float64 {
	maxLogit := float64(candidates[0].logit)
	probs := make([]float64, len(candidates))
	var total float64
	for i, candidate := range candidates {
		prob := math.Exp((float64(candidate.logit) - maxLogit) / temperature)
		probs[i] = prob
		total += prob
	}
	for i := range probs {
		probs[i] /= total
	}
	return probs
}

func applyTopP(candidates []sampleCandidate, probs []float64, topP float64) ([]sampleCandidate, []float64) {
	var cumulative float64
	keep := len(candidates)
	for i, prob := range probs {
		cumulative += prob
		if cumulative >= topP {
			keep = i + 1
			break
		}
	}
	candidates = candidates[:keep]
	probs = probs[:keep]
	var total float64
	for _, prob := range probs {
		total += prob
	}
	for i := range probs {
		probs[i] /= total
	}
	return candidates, probs
}

func sampleCandidates(rng *rand.Rand, candidates []sampleCandidate, probs []float64) int {
	draw := rng.Float64()
	var cumulative float64
	for i, prob := range probs {
		cumulative += prob
		if draw <= cumulative {
			return candidates[i].index
		}
	}
	return candidates[len(candidates)-1].index
}
