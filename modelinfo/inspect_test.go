package modelinfo

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInspectParsesMetadataAndDetectsFiles(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`)
	writeFile(t, dir, "generation_config.json", `{"temperature":0.7}`)
	writeFile(t, dir, "tokenizer.json", `{}`)
	writeFile(t, dir, "model.safetensors", "weights")
	writeFile(t, dir, "notes.txt", "ignored")

	info, err := Inspect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.ModelType != "qwen3" {
		t.Fatalf("ModelType = %q", info.ModelType)
	}
	if len(info.Architectures) != 1 || info.Architectures[0] != "Qwen3ForCausalLM" {
		t.Fatalf("Architectures = %#v", info.Architectures)
	}
	if len(info.Files) != 4 {
		t.Fatalf("Files = %#v", info.Files)
	}
	if !hasFileKind(info.Files, "tokenizer.json", "tokenizer") {
		t.Fatalf("missing tokenizer file: %#v", info.Files)
	}
	if !hasFileKind(info.Files, "model.safetensors", "safetensors") {
		t.Fatalf("missing safetensors file: %#v", info.Files)
	}
}

func TestInspectAllowsMissingOptionalJSON(t *testing.T) {
	info, err := Inspect(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if info.Config != nil || info.GenerationConfig != nil || len(info.Files) != 0 {
		t.Fatalf("unexpected info: %#v", info)
	}
}

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	path := filepath.Join(dir, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func hasFileKind(files []File, path, kind string) bool {
	for _, file := range files {
		if file.Path == path && file.Kind == kind {
			return true
		}
	}
	return false
}
