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

func testGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 1, 5)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFUint32KV(t, &buf, "llama.context_length", 4096)
	writeGGUFStringKV(t, &buf, "tokenizer.chat_template", "[INST] {{ message }} [/INST]")
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	return buf.Bytes()
}

func backendCompatible(backends []BackendCompatibility, name string) bool {
	for _, backend := range backends {
		if backend.Name == name {
			return backend.Compatible
		}
	}
	return false
}
