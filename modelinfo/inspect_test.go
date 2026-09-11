package modelinfo

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInspectParsesMetadataAndDetectsFiles(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`)
	writeFile(t, dir, "generation_config.json", `{"temperature":0.7,"top_p":0.8,"eos_token_id":[151645,151643]}`)
	writeFile(t, dir, "tokenizer.json", `{}`)
	writeFile(t, dir, "model.safetensors", "weights")
	writeFile(t, dir, "README.md", strings.Join([]string{
		"---",
		"license: apache-2.0",
		"pipeline_tag: text-generation",
		"library_name: transformers",
		"language: [en, id]",
		"tags:",
		"- qwen",
		"- causal-lm",
		"datasets:",
		"- fineweb",
		"base_model: Qwen/Qwen3-0.6B",
		"---",
		"# Qwen3 0.6B",
	}, "\n"))
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
	if len(info.Files) != 5 {
		t.Fatalf("Files = %#v", info.Files)
	}
	if !hasFileKind(info.Files, "tokenizer.json", "tokenizer") {
		t.Fatalf("missing tokenizer file: %#v", info.Files)
	}
	if !hasFileKind(info.Files, "model.safetensors", "safetensors") {
		t.Fatalf("missing safetensors file: %#v", info.Files)
	}
	if !hasFileKind(info.Files, "README.md", "model_card") {
		t.Fatalf("missing model card file: %#v", info.Files)
	}
	if info.Card == nil || info.Card.Title != "Qwen3 0.6B" || info.Card.License != "apache-2.0" {
		t.Fatalf("Card = %#v", info.Card)
	}
	if len(info.Card.Tags) != 2 || info.Card.Tags[0] != "qwen" || len(info.Card.Languages) != 2 {
		t.Fatalf("Card lists = %#v", info.Card)
	}
	if info.Generation == nil || info.Generation.Temperature == nil || *info.Generation.Temperature != 0.7 {
		t.Fatalf("Generation = %#v", info.Generation)
	}
	if len(info.Generation.EOSTokenIDs) != 2 || info.Generation.EOSTokenIDs[0] != 151645 {
		t.Fatalf("EOSTokenIDs = %#v", info.Generation.EOSTokenIDs)
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

func TestInspectModelCardFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "README.md")
	if err := os.WriteFile(path, []byte("---\nlicense: mit\n---\n# Card\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := Inspect(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Card == nil || info.Card.Title != "Card" || info.Card.License != "mit" {
		t.Fatalf("Card = %#v", info.Card)
	}
	if !hasFileKind(info.Files, "README.md", "model_card") {
		t.Fatalf("Files = %#v", info.Files)
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
