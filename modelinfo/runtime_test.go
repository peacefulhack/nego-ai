package modelinfo

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveRuntimeFileAcceptsFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ResolveRuntimeFile(path, "gguf")
	if err != nil {
		t.Fatal(err)
	}
	if got != path {
		t.Fatalf("ResolveRuntimeFile = %q, want %q", got, path)
	}
}

func TestResolveRuntimeFileChoosesBestCandidate(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "nested/small.gguf", "small")
	writeFile(t, dir, "model.gguf", strings.Repeat("x", 10))
	writeFile(t, dir, "other.onnx", strings.Repeat("x", 100))

	got, err := ResolveRuntimeFile(dir, "gguf")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(dir, "model.gguf")
	if got != want {
		t.Fatalf("ResolveRuntimeFile = %q, want %q", got, want)
	}
}

func TestResolveRuntimeFileReportsMissingCandidate(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{}`)
	_, err := ResolveRuntimeFile(dir, "gguf")
	if err == nil {
		t.Fatal("expected missing runtime file error")
	}
	if !strings.Contains(err.Error(), "no supported runtime file") {
		t.Fatalf("unexpected error: %v", err)
	}
}
