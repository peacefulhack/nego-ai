package convert

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestConvertGGUFUsesExternalConverter(t *testing.T) {
	out := filepath.Join(t.TempDir(), "model.gguf")
	err := ConvertGGUF(context.Background(), GGUFOptions{
		Converter: fakeConverter(t),
		ModelDir:  "model-dir",
		Output:    out,
		Quantize:  "q8_0",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatal(err)
	}
}

func TestConvertGGUFUsesEnvConverter(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CONVERTER", fakeConverter(t))
	out := filepath.Join(t.TempDir(), "nested", "model.gguf")
	err := ConvertGGUF(context.Background(), GGUFOptions{
		ModelDir: "model-dir",
		Output:   out,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatal(err)
	}
}

func TestConvertGGUFValidatesOptions(t *testing.T) {
	if err := ConvertGGUF(context.Background(), GGUFOptions{}); err == nil {
		t.Fatal("expected validation error")
	}
}

func TestConvertGGUFExplainsMissingNativeConverter(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CONVERTER", "")
	t.Setenv("NEGO_LLAMA_CONVERT", "")
	err := ConvertGGUF(context.Background(), GGUFOptions{ModelDir: "model-dir", Output: "model.gguf"})
	if err == nil {
		t.Fatal("expected converter error")
	}
	if !strings.Contains(err.Error(), "does not natively convert") || !strings.Contains(err.Error(), "NEGO_LLAMA_CONVERTER") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func fakeConverter(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, "convert.bat")
		if err := os.WriteFile(path, []byte("@echo off\n:loop\nif \"%1\"==\"--outfile\" (echo gguf > %2 & exit /b 0)\nshift\ngoto loop\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	path := filepath.Join(dir, "convert.sh")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nwhile [ \"$1\" != \"\" ]; do if [ \"$1\" = \"--outfile\" ]; then echo gguf > \"$2\"; exit 0; fi; shift; done\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}
