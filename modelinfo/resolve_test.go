package modelinfo

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveGGUFArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, testGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}

	artifact, err := Resolve(path)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Format != ArtifactFormatGGUF {
		t.Fatalf("Format = %q", artifact.Format)
	}
	if artifact.RecommendedRunBackend != "native" {
		t.Fatalf("RecommendedRunBackend = %q", artifact.RecommendedRunBackend)
	}
	if artifact.RecommendedTrainBackend != "native-token-bias" {
		t.Fatalf("RecommendedTrainBackend = %q", artifact.RecommendedTrainBackend)
	}
	if artifact.ParameterCount != 32 {
		t.Fatalf("ParameterCount = %d", artifact.ParameterCount)
	}
	if !capabilityStatus(artifact.RunBackends, "native", CapabilityExperimental) {
		t.Fatalf("unexpected run backends: %#v", artifact.RunBackends)
	}
}

func TestResolveHFSafetensorsArtifact(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`)
	writeFile(t, dir, "tokenizer.json", `{}`)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsFixture(t, `{
		"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}
	}`, 8), 0o644); err != nil {
		t.Fatal(err)
	}

	artifact, err := Resolve(dir)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Format != ArtifactFormatHFSafetensors {
		t.Fatalf("Format = %q", artifact.Format)
	}
	if artifact.RecommendedRunBackend != "native-hf" {
		t.Fatalf("RecommendedRunBackend = %q", artifact.RecommendedRunBackend)
	}
	if artifact.RecommendedTrainBackend != "native-token-bias" {
		t.Fatalf("RecommendedTrainBackend = %q", artifact.RecommendedTrainBackend)
	}
	if artifact.ParameterCount != 4 || artifact.TensorBytes != 8 {
		t.Fatalf("unexpected tensor summary: %#v", artifact)
	}
	if !capabilityStatus(artifact.RunBackends, "native-hf", CapabilityExperimental) {
		t.Fatalf("unexpected run backends: %#v", artifact.RunBackends)
	}
	if !capabilityStatus(artifact.TrainBackends, "native-token-bias", CapabilityExperimental) {
		t.Fatalf("unexpected train backends: %#v", artifact.TrainBackends)
	}
}

func capabilityStatus(capabilities []ArtifactCapability, name string, status ArtifactCapabilityStatus) bool {
	for _, capability := range capabilities {
		if capability.Name == name && capability.Status == status {
			return true
		}
	}
	return false
}
