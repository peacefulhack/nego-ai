package convert

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
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

func TestConvertGGUFValidatesOptions(t *testing.T) {
	if err := ConvertGGUF(context.Background(), GGUFOptions{}); err == nil {
		t.Fatal("expected validation error")
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
