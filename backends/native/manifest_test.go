package native

import (
	"context"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

func TestBuildTensorManifestReportsMissingAndTiedOutput(t *testing.T) {
	spec := ModelSpec{
		Architecture:       "llama",
		EmbeddingLength:    4,
		BlockCount:         1,
		FeedForwardLength:  8,
		AttentionHeadCount: 2,
		KVHeadCount:        1,
		RopeTheta:          10000,
	}
	names := tensorNames(spec)
	info := &modelinfo.GGUFInfo{
		Tensors: []modelinfo.GGUFTensor{
			{Name: names.TokenEmbedding, Shape: []uint64{4, 16}},
			{Name: names.OutputNorm, Shape: []uint64{4}},
		},
	}
	report := buildTensorManifest(info, spec, names)
	if !report.TiedOutput {
		t.Fatal("expected tied output when output.weight is missing")
	}
	if report.Ready() {
		t.Fatal("expected manifest to be unready with missing block tensors")
	}
	if len(report.Missing) == 0 || !containsString(report.Missing, names.Blocks[0].AttentionQ) {
		t.Fatalf("expected missing block tensors, got %#v", report.Missing)
	}
	if !containsString(report.Optional, names.Blocks[0].AttentionQNorm) || !containsString(report.Optional, names.Blocks[0].AttentionKNorm) {
		t.Fatalf("expected optional q/k norms, got %#v", report.Optional)
	}
	if len(report.MissingShape) != 0 {
		t.Fatalf("unexpected shape errors: %#v", report.MissingShape)
	}
}

func TestBuildTensorManifestReportsShapeMismatch(t *testing.T) {
	spec := ModelSpec{
		Architecture:       "llama",
		EmbeddingLength:    4,
		BlockCount:         1,
		FeedForwardLength:  8,
		AttentionHeadCount: 2,
		KVHeadCount:        1,
		RopeTheta:          10000,
	}
	names := tensorNames(spec)
	info := &modelinfo.GGUFInfo{
		Tensors: []modelinfo.GGUFTensor{
			{Name: names.Blocks[0].AttentionQ, Shape: []uint64{4, 5}},
			{Name: names.Blocks[0].AttentionQNorm, Shape: []uint64{1}},
		},
	}
	report := buildTensorManifest(info, spec, names)
	if len(report.MissingShape) != 2 ||
		!containsStringWith(report.MissingShape, names.Blocks[0].AttentionQ) ||
		!containsStringWith(report.MissingShape, names.Blocks[0].AttentionQNorm) {
		t.Fatalf("unexpected shape errors: %#v", report.MissingShape)
	}
}

func TestBuildTensorManifestUsesAttentionKeyLengthMetadata(t *testing.T) {
	spec := ModelSpec{
		Architecture:         "qwen2",
		EmbeddingLength:      10,
		BlockCount:           1,
		FeedForwardLength:    20,
		AttentionHeadCount:   3,
		KVHeadCount:          1,
		AttentionKeyLength:   4,
		AttentionValueLength: 4,
		RopeTheta:            1000000,
	}
	names := tensorNames(spec)
	info := &modelinfo.GGUFInfo{
		Tensors: []modelinfo.GGUFTensor{
			{Name: names.TokenEmbedding, Shape: []uint64{10, 16}},
			{Name: names.OutputNorm, Shape: []uint64{10}},
			{Name: names.Blocks[0].AttentionNorm, Shape: []uint64{10}},
			{Name: names.Blocks[0].AttentionQ, Shape: []uint64{10, 12}},
			{Name: names.Blocks[0].AttentionQNorm, Shape: []uint64{4}},
			{Name: names.Blocks[0].AttentionK, Shape: []uint64{10, 4}},
			{Name: names.Blocks[0].AttentionKNorm, Shape: []uint64{4}},
			{Name: names.Blocks[0].AttentionV, Shape: []uint64{10, 4}},
			{Name: names.Blocks[0].AttentionOut, Shape: []uint64{12, 10}},
			{Name: names.Blocks[0].FFNNorm, Shape: []uint64{10}},
			{Name: names.Blocks[0].FFNGate, Shape: []uint64{10, 20}},
			{Name: names.Blocks[0].FFNUp, Shape: []uint64{10, 20}},
			{Name: names.Blocks[0].FFNDown, Shape: []uint64{20, 10}},
		},
	}
	report := buildTensorManifest(info, spec, names)
	if !report.Ready() {
		t.Fatalf("expected Qwen-style manifest to be ready: %#v", report)
	}
}

func TestModelExposesTensorManifest(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t), Options: allowIncompleteOptions()})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	report := nativeModel.TensorManifest()
	if !containsString(report.Required, "token_embd.weight") {
		t.Fatalf("unexpected manifest: %#v", report)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func containsStringWith(values []string, want string) bool {
	for _, value := range values {
		if strings.Contains(value, want) {
			return true
		}
	}
	return false
}
