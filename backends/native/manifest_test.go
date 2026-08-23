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
		},
	}
	report := buildTensorManifest(info, spec, names)
	if len(report.MissingShape) != 1 || !strings.Contains(report.MissingShape[0], names.Blocks[0].AttentionQ) {
		t.Fatalf("unexpected shape errors: %#v", report.MissingShape)
	}
}

func TestModelExposesTensorManifest(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
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
