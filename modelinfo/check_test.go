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
	if len(native.UnsupportedTensorTypes) != 1 || native.UnsupportedTensorTypes[0] != "q2_k" {
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
	writeGGUFTensor(t, &buf, "blk.0.attn_q.weight", []uint64{16, 16}, 10, 0)
	return buf.Bytes()
}

func testNativeKQuantGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 2, 3)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeGGUFTensor(t, &buf, "blk.0.attn_q.weight", []uint64{16, 16}, 12, 0)
	writeGGUFTensor(t, &buf, "blk.0.ffn_gate.weight", []uint64{16, 16}, 14, 144)
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
