package modelinfo

import (
	"fmt"
	"sort"
)

type HFShapeReport struct {
	Ready        bool     `json:"ready"`
	Checked      []string `json:"checked,omitempty"`
	MissingShape []string `json:"missing_shape,omitempty"`
}

func BuildHFShapeReport(path string) (*HFShapeReport, error) {
	info, err := Inspect(path)
	if err != nil {
		return nil, err
	}
	return HFShapeReportFromInfo(info)
}

func HFShapeReportFromInfo(info *Info) (*HFShapeReport, error) {
	if info == nil {
		return nil, fmt.Errorf("model info is nil")
	}
	if info.Safetensors == nil {
		return nil, fmt.Errorf("model has no safetensors metadata")
	}
	if info.HFSpec == nil {
		return nil, fmt.Errorf("model has no Hugging Face model spec")
	}
	if info.HFWeights == nil {
		return nil, fmt.Errorf("model has no Hugging Face weight manifest")
	}
	report := &HFShapeReport{}
	available := make(map[string]SafetensorsTensor, len(info.Safetensors.Tensors))
	for _, tensor := range info.Safetensors.Tensors {
		available[tensor.Name] = tensor
	}
	expect := func(name string, shape ...uint64) {
		if name == "" {
			return
		}
		tensor, ok := available[name]
		if !ok {
			return
		}
		report.Checked = append(report.Checked, name)
		if !sameHFShape(tensor.Shape, shape) {
			report.MissingShape = append(report.MissingShape, fmt.Sprintf("%s got %v want %v", name, tensor.Shape, shape))
		}
	}
	spec := info.HFSpec
	names := info.HFWeights
	qLength := spec.AttentionHeadCount * spec.HeadDim
	kvLength := spec.KVHeadCount * spec.HeadDim
	expect(names.TokenEmbedding, spec.VocabSize, spec.EmbeddingLength)
	expect(names.OutputNorm, spec.EmbeddingLength)
	if !names.TiedOutput {
		expect(names.Output, spec.VocabSize, spec.EmbeddingLength)
	}
	for _, block := range names.Blocks {
		expect(block.InputNorm, spec.EmbeddingLength)
		expect(block.AttentionQ, qLength, spec.EmbeddingLength)
		expect(block.AttentionQNorm, spec.HeadDim)
		expect(block.AttentionK, kvLength, spec.EmbeddingLength)
		expect(block.AttentionKNorm, spec.HeadDim)
		expect(block.AttentionV, kvLength, spec.EmbeddingLength)
		expect(block.AttentionOut, spec.EmbeddingLength, qLength)
		expect(block.PostNorm, spec.EmbeddingLength)
		expect(block.FFNGate, spec.FeedForwardLength, spec.EmbeddingLength)
		expect(block.FFNUp, spec.FeedForwardLength, spec.EmbeddingLength)
		expect(block.FFNDown, spec.EmbeddingLength, spec.FeedForwardLength)
	}
	sort.Strings(report.Checked)
	sort.Strings(report.MissingShape)
	report.Ready = info.HFSpec.Ready && info.HFWeights.Ready && len(report.MissingShape) == 0
	return report, nil
}

func sameHFShape(got []uint64, want []uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
