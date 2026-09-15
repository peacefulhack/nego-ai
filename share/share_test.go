package share

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

func TestBuildManifestDetectsNativeAdapter(t *testing.T) {
	dir := t.TempDir()
	manifestJSON := `{
		"version":1,
		"type":"nego-native-adapter",
		"base_model":"./models/qwen3",
		"adapter_path":"adapter.json",
		"recommended_backend":"native-hf",
		"method":"token-bias",
		"dataset_format":"completion",
		"tokenizer":{"format":"gguf","model":"llama","pre_tokenizer":"llama-bpe","vocab_size":10},
		"vocab_size":10,
		"updated_tokens":1,
		"train_tokens":3,
		"top_tokens":[{"id":1,"text":"hello","count":3,"bias":0.1}]
	}`
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(manifestJSON), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "adapter.json"), []byte(`{"version":1,"type":"token_bias","vocab_size":10,"bias":{"1":0.1}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	manifest, err := BuildManifest(ManifestOptions{Path: dir, RepoID: "user/qwen3-adapter"})
	if err != nil {
		t.Fatal(err)
	}
	if manifest.ArtifactType != "native-adapter" || manifest.BaseModel != "./models/qwen3" || manifest.NativeAdapter == nil {
		t.Fatalf("unexpected adapter metadata: %#v", manifest)
	}
	if manifest.NativeAdapter.UpdatedTokens != 1 || len(manifest.NativeAdapter.TopTokens) != 1 {
		t.Fatalf("unexpected native adapter summary: %#v", manifest.NativeAdapter)
	}
	if manifest.NativeAdapter.Tokenizer == nil || manifest.NativeAdapter.Tokenizer.Format != "gguf" || manifest.NativeAdapter.Tokenizer.PreTokenizer != "llama-bpe" {
		t.Fatalf("unexpected tokenizer metadata: %#v", manifest.NativeAdapter.Tokenizer)
	}
	if manifest.NativeAdapter.TopTokens[0].Text != "" {
		t.Fatalf("share manifest should not include token text: %#v", manifest.NativeAdapter.TopTokens)
	}
	if !hasFileKind(manifest.Files, "manifest.json", "native_manifest") || !hasFileKind(manifest.Files, "adapter.json", "native_adapter") {
		t.Fatalf("unexpected file kinds: %#v", manifest.Files)
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

func hasFileKind(files []File, path, kind string) bool {
	for _, file := range files {
		if file.Path == path && file.Kind == kind {
			return true
		}
	}
	return false
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

func TestCheckReportsLargeFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("model card"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("large model"), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Check(CheckOptions{Path: dir, RepoID: "user/model", MaxInlineSize: 10})
	if err != nil {
		t.Fatal(err)
	}
	if report.Valid || len(report.LargeFiles) != 1 || report.LargeFiles[0].Path != "model.safetensors" || report.MaxInlineSize != 10 {
		t.Fatalf("unexpected report: %#v", report)
	}
}

func TestPackageArchiveWritesTarGzip(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "model")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "README.md"), []byte("model card"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "adapter.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "nego-share-manifest.json"), []byte("old manifest"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(modelDir, "model.tar.gz")
	manifest, err := PackageArchive(PackageOptions{
		Path:            modelDir,
		Output:          out,
		RepoID:          "user/model",
		BaseModel:       "Qwen/Qwen3-0.6B",
		IncludeManifest: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Files) != 2 {
		t.Fatalf("archive output should be excluded from manifest: %#v", manifest.Files)
	}
	names := readTarGzipNames(t, out)
	for _, want := range []string{"nego-share-manifest.json", "README.md", "adapter.safetensors"} {
		if !containsString(names, want) {
			t.Fatalf("expected %q in archive names %#v", want, names)
		}
	}
	if containsString(names, "model.tar.gz") {
		t.Fatalf("archive included itself: %#v", names)
	}
	if countString(names, "nego-share-manifest.json") != 1 {
		t.Fatalf("archive should include exactly one generated manifest: %#v", names)
	}
}

func TestBuildManifestRejectsSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation often requires Windows developer mode or elevated privileges")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "target.txt")
	link := filepath.Join(dir, "link.txt")
	if err := os.WriteFile(target, []byte("target"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	_, err := BuildManifest(ManifestOptions{Path: dir})
	if err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func readTarGzipNames(t *testing.T, path string) []string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	var names []string
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, header.Name)
	}
	return names
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func countString(values []string, want string) int {
	var count int
	for _, value := range values {
		if value == want {
			count++
		}
	}
	return count
}
