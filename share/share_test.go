package share

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildManifest(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("model card"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "adapter.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, err := BuildManifest(ManifestOptions{Path: dir, RepoID: "user/model", BaseModel: "Qwen/Qwen3-0.6B"})
	if err != nil {
		t.Fatal(err)
	}
	if manifest.RepoID != "user/model" || manifest.BaseModel != "Qwen/Qwen3-0.6B" {
		t.Fatalf("unexpected manifest metadata: %#v", manifest)
	}
	if len(manifest.Files) != 2 || manifest.TotalSize == 0 {
		t.Fatalf("unexpected files: %#v", manifest.Files)
	}
	if manifest.Files[0].Path != "README.md" || manifest.Files[0].Kind != "model_card" || manifest.Files[0].SHA256 == "" {
		t.Fatalf("unexpected first file: %#v", manifest.Files[0])
	}
}

func TestWriteManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "share-manifest.json")
	manifest := &Manifest{Path: "model", Files: []File{{Path: "README.md", Size: 1, SHA256: "abc"}}}
	if err := WriteManifest(path, manifest); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got Manifest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Files) != 1 || got.Files[0].Path != "README.md" {
		t.Fatalf("unexpected manifest: %#v", got)
	}
}

func TestBuildManifestExcludesPaths(t *testing.T) {
	dir := t.TempDir()
	oldManifest := filepath.Join(dir, "share-manifest.json")
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("model card"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(oldManifest, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, err := BuildManifest(ManifestOptions{Path: dir, Exclude: []string{oldManifest}})
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Files) != 1 || manifest.Files[0].Path != "README.md" {
		t.Fatalf("unexpected files: %#v", manifest.Files)
	}
}

func TestBuildManifestRejectsFilePath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, []byte("model"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildManifest(ManifestOptions{Path: path}); err == nil {
		t.Fatal("expected file path error")
	}
}
