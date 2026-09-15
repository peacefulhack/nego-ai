package modelinfo

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
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
	if report.Tokenizer == nil ||
		report.Tokenizer.Model != "llama" ||
		report.Tokenizer.PreTokenizer != "llama-bpe" ||
		report.Tokenizer.VocabSize != 2 {
		t.Fatalf("unexpected tokenizer report: %#v", report.Tokenizer)
	}
	if report.Tokenizer.BOSTokenID == nil ||
		*report.Tokenizer.BOSTokenID != 0 ||
		report.Tokenizer.EOSTokenID == nil ||
		*report.Tokenizer.EOSTokenID != 1 {
		t.Fatalf("unexpected tokenizer special IDs: %#v", report.Tokenizer)
	}
	if report.Tokenizer.AddBOS == nil ||
		!*report.Tokenizer.AddBOS ||
		report.Tokenizer.AddEOS == nil ||
		*report.Tokenizer.AddEOS {
		t.Fatalf("unexpected tokenizer defaults: %#v", report.Tokenizer)
	}
	if !backendCompatible(report.Backends, "llama.cpp") {
		t.Fatalf("expected llama.cpp compatibility: %#v", report.Backends)
	}
	if report.NativeReadiness == nil || !report.NativeReadiness.Ready {
		t.Fatalf("expected native readiness: %#v", report.NativeReadiness)
	}
	if !backendCompatible(report.Backends, "native") {
		t.Fatalf("expected native compatibility: %#v", report.Backends)
	}
	if report.Memory == nil || report.Memory.TotalBytes == 0 || report.Memory.RuntimeBytes == 0 {
		t.Fatalf("expected memory estimate: %#v", report.Memory)
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
	if report.NativeReadiness == nil || len(report.NativeReadiness.SpecIssues) == 0 {
		t.Fatalf("expected native readiness spec issues: %#v", report.NativeReadiness)
	}
}

func TestCheckReportsNativeUnsupportedRequiredTensorNames(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, testUnsupportedRequiredNativeGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Check(path)
	if err != nil {
		t.Fatal(err)
	}
	if report.NativeReadiness == nil || report.NativeReadiness.Ready {
		t.Fatalf("expected native unreadiness: %#v", report.NativeReadiness)
	}
	if len(report.NativeReadiness.UnsupportedRequired) != 1 ||
		report.NativeReadiness.UnsupportedRequired[0] != "blk.0.attn_q.weight:i8" {
		t.Fatalf("unexpected unsupported required tensors: %#v", report.NativeReadiness.UnsupportedRequired)
	}
	if !strings.Contains(report.NativeReadiness.Reason, "unsupported required tensor types") {
		t.Fatalf("unexpected reason: %s", report.NativeReadiness.Reason)
	}
	native, ok := backendByName(report.Backends, "native")
	if !ok || native.Compatible {
		t.Fatalf("expected native backend incompatibility: %#v", report.Backends)
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

func TestCheckReportsNativeQwenAttentionKeyLengthCompatibility(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qwen.gguf")
	if err := os.WriteFile(path, testQwenAttentionKeyLengthGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Check(path)
	if err != nil {
		t.Fatal(err)
	}
	if report.NativeReadiness == nil || !report.NativeReadiness.Ready {
		t.Fatalf("expected native readiness: %#v", report.NativeReadiness)
	}
	if !backendCompatible(report.Backends, "native") {
		t.Fatalf("expected native backend compatibility: %#v", report.Backends)
	}
}

func TestCheckReportsNativeMissingTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, testIncompleteNativeGGUF(t), 0o644); err != nil {
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
	if report.NativeReadiness == nil || report.NativeReadiness.Ready || len(report.NativeReadiness.MissingTensors) == 0 {
		t.Fatalf("expected missing tensor readiness details: %#v", report.NativeReadiness)
	}
	if !containsString(report.NativeReadiness.MissingTensors, "output_norm.weight") {
		t.Fatalf("expected output_norm.weight missing, got %#v", report.NativeReadiness.MissingTensors)
	}
}

func TestCheckReportsNativeHFCompatibility(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":4,"max_position_embeddings":8,"hidden_size":4,"num_hidden_layers":1,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2}`)
	writeFile(t, dir, "generation_config.json", `{"max_new_tokens":64,"temperature":0.6}`)
	writeFile(t, dir, "tokenizer.json", `{}`)
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
	writeGGUFHeader(t, &buf, 3, 11, 16)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFUint32KV(t, &buf, "llama.context_length", 4096)
	writeGGUFUint32KV(t, &buf, "llama.embedding_length", 4)
	writeGGUFUint32KV(t, &buf, "llama.block_count", 1)
	writeGGUFUint32KV(t, &buf, "llama.feed_forward_length", 8)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count", 2)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count_kv", 1)
	writeGGUFStringKV(t, &buf, "tokenizer.chat_template", "[INST] {{ message }} [/INST]")
	writeGGUFStringKV(t, &buf, "tokenizer.ggml.model", "llama")
	writeGGUFStringKV(t, &buf, "tokenizer.ggml.pre", "llama-bpe")
	writeGGUFUint32KV(t, &buf, "tokenizer.ggml.bos_token_id", 0)
	writeGGUFUint32KV(t, &buf, "tokenizer.ggml.eos_token_id", 1)
	writeGGUFBoolKV(t, &buf, "tokenizer.ggml.add_bos_token", true)
	writeGGUFBoolKV(t, &buf, "tokenizer.ggml.add_eos_token", false)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeTinyNativeTensorManifest(t, &buf, 1)
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

func testIncompleteNativeGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 1, 9)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 1)
	writeGGUFUint32KV(t, &buf, "llama.context_length", 128)
	writeGGUFUint32KV(t, &buf, "llama.embedding_length", 4)
	writeGGUFUint32KV(t, &buf, "llama.block_count", 1)
	writeGGUFUint32KV(t, &buf, "llama.feed_forward_length", 8)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count", 2)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count_kv", 1)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeGGUFTensor(t, &buf, "token_embd.weight", []uint64{4, 2}, 1, 0)
	return buf.Bytes()
}

func testNativeKQuantGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 11, 9)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFUint32KV(t, &buf, "llama.context_length", 4096)
	writeGGUFUint32KV(t, &buf, "llama.embedding_length", 256)
	writeGGUFUint32KV(t, &buf, "llama.block_count", 1)
	writeGGUFUint32KV(t, &buf, "llama.feed_forward_length", 512)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count", 4)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count_kv", 2)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeTinyNativeKQuantTensorManifest(t, &buf)
	return buf.Bytes()
}

func testQwenAttentionKeyLengthGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 13, 11)
	writeGGUFStringKV(t, &buf, "general.architecture", "qwen2")
	writeGGUFUint32KV(t, &buf, "general.file_type", 0)
	writeGGUFUint32KV(t, &buf, "qwen2.context_length", 128)
	writeGGUFUint32KV(t, &buf, "qwen2.embedding_length", 10)
	writeGGUFUint32KV(t, &buf, "qwen2.block_count", 1)
	writeGGUFUint32KV(t, &buf, "qwen2.feed_forward_length", 20)
	writeGGUFUint32KV(t, &buf, "qwen2.attention.head_count", 3)
	writeGGUFUint32KV(t, &buf, "qwen2.attention.head_count_kv", 1)
	writeGGUFUint32KV(t, &buf, "qwen2.attention.key_length", 4)
	writeGGUFUint32KV(t, &buf, "qwen2.attention.value_length", 4)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeTinyQwenNativeTensorManifest(t, &buf, 0)
	return buf.Bytes()
}

func testUnsupportedRequiredNativeGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 11, 9)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 0)
	writeGGUFUint32KV(t, &buf, "llama.context_length", 4096)
	writeGGUFUint32KV(t, &buf, "llama.embedding_length", 4)
	writeGGUFUint32KV(t, &buf, "llama.block_count", 1)
	writeGGUFUint32KV(t, &buf, "llama.feed_forward_length", 8)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count", 2)
	writeGGUFUint32KV(t, &buf, "llama.attention.head_count_kv", 1)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeTinyNativeTensorManifestWithTypeOverride(t, &buf, "blk.0.attn_q.weight", 24)
	return buf.Bytes()
}

func writeTinyQwenNativeTensorManifest(t *testing.T, buf *bytes.Buffer, typ uint32) {
	t.Helper()
	writeGGUFTensor(t, buf, "token_embd.weight", []uint64{10, 2}, typ, 0)
	writeGGUFTensor(t, buf, "output_norm.weight", []uint64{10}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_norm.weight", []uint64{10}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_q.weight", []uint64{10, 12}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_q_norm.weight", []uint64{4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_k.weight", []uint64{10, 4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_k_norm.weight", []uint64{4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_v.weight", []uint64{10, 4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_output.weight", []uint64{12, 10}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_norm.weight", []uint64{10}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_gate.weight", []uint64{10, 20}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_up.weight", []uint64{10, 20}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_down.weight", []uint64{20, 10}, typ, 0)
}

func writeTinyNativeTensorManifest(t *testing.T, buf *bytes.Buffer, typ uint32) {
	t.Helper()
	writeGGUFTensor(t, buf, "token_embd.weight", []uint64{4, 2}, typ, 0)
	writeGGUFTensor(t, buf, "output_norm.weight", []uint64{4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_norm.weight", []uint64{4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_q.weight", []uint64{4, 4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_k.weight", []uint64{4, 2}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_v.weight", []uint64{4, 2}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_output.weight", []uint64{4, 4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_norm.weight", []uint64{4}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_gate.weight", []uint64{4, 8}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_up.weight", []uint64{4, 8}, typ, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_down.weight", []uint64{8, 4}, typ, 0)
}

func writeTinyNativeTensorManifestWithTypeOverride(t *testing.T, buf *bytes.Buffer, overrideName string, overrideType uint32) {
	t.Helper()
	writeTensor := func(name string, shape []uint64) {
		typ := uint32(0)
		if name == overrideName {
			typ = overrideType
		}
		writeGGUFTensor(t, buf, name, shape, typ, 0)
	}
	writeTensor("token_embd.weight", []uint64{4, 2})
	writeTensor("output_norm.weight", []uint64{4})
	writeTensor("blk.0.attn_norm.weight", []uint64{4})
	writeTensor("blk.0.attn_q.weight", []uint64{4, 4})
	writeTensor("blk.0.attn_k.weight", []uint64{4, 2})
	writeTensor("blk.0.attn_v.weight", []uint64{4, 2})
	writeTensor("blk.0.attn_output.weight", []uint64{4, 4})
	writeTensor("blk.0.ffn_norm.weight", []uint64{4})
	writeTensor("blk.0.ffn_gate.weight", []uint64{4, 8})
	writeTensor("blk.0.ffn_up.weight", []uint64{4, 8})
	writeTensor("blk.0.ffn_down.weight", []uint64{8, 4})
}

func writeTinyNativeKQuantTensorManifest(t *testing.T, buf *bytes.Buffer) {
	t.Helper()
	writeGGUFTensor(t, buf, "token_embd.weight", []uint64{256, 2}, 10, 0)
	writeGGUFTensor(t, buf, "output_norm.weight", []uint64{256}, 11, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_norm.weight", []uint64{256}, 12, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_q.weight", []uint64{256, 256}, 10, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_k.weight", []uint64{256, 128}, 11, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_v.weight", []uint64{256, 128}, 12, 0)
	writeGGUFTensor(t, buf, "blk.0.attn_output.weight", []uint64{256, 256}, 13, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_norm.weight", []uint64{256}, 14, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_gate.weight", []uint64{256, 512}, 10, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_up.weight", []uint64{256, 512}, 11, 0)
	writeGGUFTensor(t, buf, "blk.0.ffn_down.weight", []uint64{512, 256}, 12, 0)
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

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
