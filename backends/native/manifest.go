package native

import (
	"fmt"
	"sort"

	"github.com/gakon/nego-ai/modelinfo"
)

type TensorManifestReport struct {
	Required     []string
	Missing      []string
	Optional     []string
	MissingShape []string
	TiedOutput   bool
}

func (r TensorManifestReport) Ready() bool {
	return len(r.Missing) == 0 && len(r.MissingShape) == 0
}

func buildTensorManifest(info *modelinfo.GGUFInfo, spec ModelSpec, names TensorNames) TensorManifestReport {
	available := make(map[string]modelinfo.GGUFTensor, len(info.Tensors))
	for _, tensor := range info.Tensors {
		available[tensor.Name] = tensor
	}
	report := TensorManifestReport{
		Required: requiredTensorNames(names),
		Optional: []string{names.Output},
	}
	for _, name := range report.Required {
		if _, ok := available[name]; !ok {
			report.Missing = append(report.Missing, name)
		}
	}
	if _, ok := available[names.Output]; ok {
		report.Required = append(report.Required, names.Output)
	} else {
		report.TiedOutput = true
	}
	report.MissingShape = append(report.MissingShape, validateKnownTensorShapes(available, spec, names)...)
	sort.Strings(report.Required)
	sort.Strings(report.Missing)
	sort.Strings(report.Optional)
	sort.Strings(report.MissingShape)
	return report
}

func requiredTensorNames(names TensorNames) []string {
	out := []string{
		names.TokenEmbedding,
		names.OutputNorm,
	}
	for _, block := range names.Blocks {
		out = append(out,
			block.AttentionNorm,
			block.AttentionQ,
			block.AttentionK,
			block.AttentionV,
			block.AttentionOut,
			block.FFNNorm,
			block.FFNGate,
			block.FFNUp,
			block.FFNDown,
		)
	}
	return out
}

func validateKnownTensorShapes(available map[string]modelinfo.GGUFTensor, spec ModelSpec, names TensorNames) []string {
	var out []string
	expect := func(name string, shape ...uint64) {
		tensor, ok := available[name]
		if !ok {
			return
		}
		if !sameShape(tensor.Shape, shape) {
			out = append(out, fmt.Sprintf("%s got %v want %v", name, tensor.Shape, shape))
		}
	}
	expect(names.TokenEmbedding, spec.EmbeddingLength, 0)
	expect(names.OutputNorm, spec.EmbeddingLength)
	if _, ok := available[names.Output]; ok {
		expect(names.Output, spec.EmbeddingLength, 0)
	}
	for _, block := range names.Blocks {
		expect(block.AttentionNorm, spec.EmbeddingLength)
		expect(block.AttentionQ, spec.EmbeddingLength, spec.EmbeddingLength)
		expect(block.AttentionK, spec.EmbeddingLength, kvProjectionLength(spec))
		expect(block.AttentionV, spec.EmbeddingLength, kvProjectionLength(spec))
		expect(block.AttentionOut, spec.EmbeddingLength, spec.EmbeddingLength)
		expect(block.FFNNorm, spec.EmbeddingLength)
		expect(block.FFNGate, spec.EmbeddingLength, spec.FeedForwardLength)
		expect(block.FFNUp, spec.EmbeddingLength, spec.FeedForwardLength)
		expect(block.FFNDown, spec.FeedForwardLength, spec.EmbeddingLength)
	}
	return out
}

func sameShape(got []uint64, want []uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if want[i] != 0 && got[i] != want[i] {
			return false
		}
	}
	return true
}

func kvProjectionLength(spec ModelSpec) uint64 {
	return spec.EmbeddingLength / spec.AttentionHeadCount * spec.KVHeadCount
}
