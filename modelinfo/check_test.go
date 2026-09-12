package modelinfo

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestCheckReportsGGUFCompatibility(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, testGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Check(path)
	if err != nil {
		t.Fatal(err)
	}
	if report.RuntimeFile == nil || report.RuntimeFile.Kind != "gguf" {
		t.Fatalf("RuntimeFile = %#v", report.RuntimeFile)
	}
	if !report.ChatTemplate || report.ContextLength != 4096 || report.Quantization != "mostly_q4_k_m" {
		t.Fatalf("unexpected report: %#v", report)
	}
	if !backendCompatible(report.Backends, "llama.cpp") {
		t.Fatalf("expected llama.cpp compatibility: %#v", report.Backends)
	}
}

func TestCheckReportsNativeUnsupportedTensorTypes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, testUnsupportedNativeGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Check(path)
	if err != nil {
		t.Fatal(err)
	}
	native, ok := backendByName(report.Backends, "native")
	if !ok {
		t.Fatalf("native backend missing: %#v", report.Backends)
	}
	if native.Compatible {
		t.Fatalf("expected native incompatibility: %#v", native)
	}
	if len(native.UnsupportedTensorTypes) != 1 || native.UnsupportedTensorTypes[0] != "i8" {
		t.Fatalf("unexpected unsupported types: %#v", native.UnsupportedTensorTypes)
	}
	if len(report.Warnings) == 0 {
		t.Fatalf("expected warning: %#v", report)
	}
}

func TestCheckReportsNativeKQuantCompatibility(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, testNativeKQuantGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Check(path)
	if err != nil {
		t.Fatal(err)
	}
	native, ok := backendByName(report.Backends, "native")
	if !ok {
		t.Fatalf("native backend missing: %#v", report.Backends)
	}
	if !native.Compatible {
		t.Fatalf("expected native compatibility: %#v", native)
	}
	if len(native.UnsupportedTensorTypes) != 0 {
		t.Fatalf("unexpected unsupported types: %#v", native.UnsupportedTensorTypes)
	}
}

func TestCheckReportsNativeHFCompatibility(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":4,"max_position_embeddings":8,"hidden_size":4,"num_hidden_layers":1,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2}`)
	writeFile(t, dir, "generation_config.json", `{"max_new_tokens":64,"temperature":0.6}`)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsShapeFixture(t, hfShapeFixtureTensors(nil)), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Check(dir)
	if err != nil {
		t.Fatal(err)
	}
	nativeHF, ok := backendByName(report.Backends, "native-hf")
	if !ok {
		t.Fatalf("native-hf backend missing: %#v", report.Backends)
	}
	if !nativeHF.Compatible {
		t.Fatalf("expected native-hf compatibility: %#v", nativeHF)
	}
	if report.Generation == nil || report.Generation.MaxNewTokens == nil || *report.Generation.MaxNewTokens != 64 {
		t.Fatalf("Generation = %#v", report.Generation)
	}
}

func TestCheckReportsNativeAdapterArtifact(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "manifest.json", `{"version":1,"type":"nego-native-adapter","base_model":"./models/qwen3","adapter_path":"./outputs/qwen3/adapter.json","recommended_backend":"native-hf","method":"token-bias","updated_tokens":2}`)
	report, err := Check(dir)
	if err != nil {
		t.Fatal(err)
	}
	if report.NativeAdapter == nil || report.NativeAdapter.RecommendedBackend != "native-hf" {
		t.Fatalf("NativeAdapter = %#v", report.NativeAdapter)
	}
	if report.Artifact == nil || report.Artifact.Format != ArtifactFormatNativeAdapter || report.Artifact.RecommendedRunBackend != "native-hf" {
		t.Fatalf("Artifact = %#v", report.Artifact)
	}
	if !backendCompatible(report.Backends, "native-hf") {
		t.Fatalf("Backends = %#v", report.Backends)
	}
	if len(report.Warnings) != 0 {
		t.Fatalf("Warnings = %#v", report.Warnings)
	}
}

func testGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 1, 5)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFUint32KV(t, &buf, "llama.context_length", 4096)
	writeGGUFStringKV(t, &buf, "tokenizer.chat_template", "[INST] {{ message }} [/INST]")
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeGGUFTensor(t, &buf, "token_embd.weight", []uint64{2, 16}, 1, 0)
	return buf.Bytes()
}

func testUnsupportedNativeGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 1, 3)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeGGUFTensor(t, &buf, "blk.0.attn_q.weight", []uint64{1}, 24, 0)
	return buf.Bytes()
}

func testNativeKQuantGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 5, 3)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeGGUFTensor(t, &buf, "blk.0.attn_q.weight", []uint64{16, 16}, 10, 0)
	writeGGUFTensor(t, &buf, "blk.0.attn_k.weight", []uint64{16, 16}, 11, 84)
	writeGGUFTensor(t, &buf, "blk.0.attn_v.weight", []uint64{16, 16}, 12, 194)
	writeGGUFTensor(t, &buf, "blk.0.attn_output.weight", []uint64{16, 16}, 13, 338)
	writeGGUFTensor(t, &buf, "blk.0.ffn_gate.weight", []uint64{16, 16}, 14, 514)
	return buf.Bytes()
}

func backendCompatible(backends []BackendCompatibility, name string) bool {
	backend, ok := backendByName(backends, name)
	return ok && backend.Compatible
}

func backendByName(backends []BackendCompatibility, name string) (BackendCompatibility, bool) {
	for _, backend := range backends {
		if backend.Name == name {
			return backend, true
		}
	}
	return BackendCompatibility{}, false
}
