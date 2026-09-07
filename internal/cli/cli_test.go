package cli

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/hub"
	"github.com/gakon/nego-ai/internal/registry"
	"github.com/gakon/nego-ai/training"
)

func TestSplitFlagsAllowsFlagsAfterPositionals(t *testing.T) {
	flags, positionals := splitFlags([]string{
		"Qwen/Qwen3-0.6B",
		"--local-dir",
		"./models/qwen3",
	})
	if !reflect.DeepEqual(flags, []string{"--local-dir", "./models/qwen3"}) {
		t.Fatalf("flags = %#v", flags)
	}
	if !reflect.DeepEqual(positionals, []string{"Qwen/Qwen3-0.6B"}) {
		t.Fatalf("positionals = %#v", positionals)
	}
}

func TestSplitFlagsAllowsFilenameAndFilters(t *testing.T) {
	flags, positionals := splitFlags([]string{
		"Qwen/Qwen3-0.6B",
		"tokenizer.json",
		"--include",
		"*.json",
		"--quiet",
	})
	if !reflect.DeepEqual(flags, []string{"--include", "*.json", "--quiet"}) {
		t.Fatalf("flags = %#v", flags)
	}
	if !reflect.DeepEqual(positionals, []string{"Qwen/Qwen3-0.6B", "tokenizer.json"}) {
		t.Fatalf("positionals = %#v", positionals)
	}
}

func TestSplitFlagsTreatsGGUFAsBool(t *testing.T) {
	flags, positionals := splitFlags([]string{
		"Qwen/Qwen3-0.6B",
		"--gguf",
		"--local-dir",
		"./models/qwen3-gguf",
	})
	if !reflect.DeepEqual(flags, []string{"--gguf", "--local-dir", "./models/qwen3-gguf"}) {
		t.Fatalf("flags = %#v", flags)
	}
	if !reflect.DeepEqual(positionals, []string{"Qwen/Qwen3-0.6B"}) {
		t.Fatalf("positionals = %#v", positionals)
	}
}

func TestRenderDownloadLineDoesNotDuplicateUnits(t *testing.T) {
	line := renderDownloadLine(hub.ProgressEvent{
		Filename:   "model.safetensors",
		BytesDone:  1503238554,
		BytesTotal: 1503238554,
	}, time.Now().Add(-time.Second))

	if !strings.Contains(line, "1.4 GiB/1.4 GiB") {
		t.Fatalf("expected formatted done/total bytes, got %q", line)
	}
	if strings.HasSuffix(line, "GiBGiB") {
		t.Fatalf("line contains stale unit suffix: %q", line)
	}
}

func TestProgressBarComplete(t *testing.T) {
	if got := progressBar(10, 10, 8); got != "[========]" {
		t.Fatalf("progressBar = %q", got)
	}
}

func TestRenderProgressLinesListsFilesAndDoneState(t *testing.T) {
	entries := map[string]*progressEntry{
		"config.json": {
			name:       "config.json",
			state:      "Done",
			bytesTotal: 512,
		},
		"model.safetensors": {
			name:       "model.safetensors",
			state:      "Queued",
			bytesTotal: 1503238554,
		},
	}
	lines := renderProgressLines("Downloading 2 files", []string{"config.json", "model.safetensors"}, entries, spinnerFrame(0))
	if len(lines) != 3 {
		t.Fatalf("expected title plus two file lines, got %#v", lines)
	}
	if !strings.Contains(lines[1], "config.json") || !strings.Contains(lines[1], "Done") || !strings.Contains(lines[1], "✓") {
		t.Fatalf("expected done line for config.json, got %q", lines[1])
	}
	if !strings.Contains(lines[2], "model.safetensors") || !strings.Contains(lines[2], "Queued") {
		t.Fatalf("expected queued line for model.safetensors, got %q", lines[2])
	}
}

func TestSpinnerFrameCycles(t *testing.T) {
	want := []string{"—", "\\", "|", "/", "—"}
	for i, expected := range want {
		if got := spinnerFrame(i); got != expected {
			t.Fatalf("spinnerFrame(%d) = %q, want %q", i, got, expected)
		}
	}
}

func TestRenderProgressEntryShowsMaterializing(t *testing.T) {
	entry := &progressEntry{
		name:       "model.safetensors",
		state:      "Materializing",
		bytesDone:  536870912,
		bytesTotal: 1073741824,
		started:    time.Now().Add(-time.Second),
	}
	line := renderProgressEntry(entry, "|")
	if !strings.Contains(line, "Materializing") || !strings.Contains(line, "512.0 MiB/1.0 GiB") {
		t.Fatalf("unexpected materializing line: %q", line)
	}
}

func TestDownloadRuntimeWarningForSafetensorsOnly(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	warning := downloadRuntimeWarning(dir)
	if !strings.Contains(warning, "cannot run this directly") || !strings.Contains(warning, "--gguf") {
		t.Fatalf("unexpected warning: %q", warning)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	if warning := downloadRuntimeWarning(dir); warning != "" {
		t.Fatalf("unexpected warning with GGUF present: %q", warning)
	}
}

func TestModelsListShowsRegistryEntries(t *testing.T) {
	cacheDir := t.TempDir()
	err := registry.NewStore(cacheDir).Upsert(registry.Entry{
		RepoID:       "Qwen/Qwen3-0.6B",
		RepoType:     "model",
		Revision:     "main",
		Commit:       "abc123",
		LocalDir:     "./models/qwen3",
		SnapshotPath: cacheDir + "/models--Qwen--Qwen3-0.6B/snapshots/abc123",
		FileCount:    3,
		TotalSize:    1024,
	})
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"models", "list", "--cache-dir", cacheDir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Qwen/Qwen3-0.6B") || !strings.Contains(out, "./models/qwen3") || !strings.Contains(out, "1.0 KiB") {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestModelsListJSON(t *testing.T) {
	cacheDir := t.TempDir()
	err := registry.NewStore(cacheDir).Upsert(registry.Entry{
		RepoID:       "Qwen/Qwen3-0.6B",
		RepoType:     "model",
		Revision:     "main",
		Commit:       "abc123",
		SnapshotPath: cacheDir + "/snapshot",
		FileCount:    1,
	})
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"models", "list", "--cache-dir", cacheDir, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var entries []registry.Entry
	if err := json.Unmarshal(stdout.Bytes(), &entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].RepoID != "Qwen/Qwen3-0.6B" {
		t.Fatalf("unexpected entries: %#v", entries)
	}
}

func TestModelsListEmpty(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"models", "list", "--cache-dir", t.TempDir()}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "No local models found.") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestCacheUsageCommand(t *testing.T) {
	cacheDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(cacheDir, "file.bin"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := registry.NewStore(cacheDir).Upsert(registry.Entry{RepoID: "Qwen/Qwen3", RepoType: "model", Revision: "main", SnapshotPath: cacheDir}); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"cache", "usage", "--cache-dir", cacheDir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Registered models: 1") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestCacheGCRequiresYes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"cache", "gc", "--cache-dir", t.TempDir()}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func TestCacheGCRemovesTemporaryFiles(t *testing.T) {
	cacheDir := t.TempDir()
	tmpPath := filepath.Join(cacheDir, ".nego-temp")
	keepPath := filepath.Join(cacheDir, "blob")
	if err := os.WriteFile(tmpPath, []byte("temp"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keepPath, []byte("blob"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"cache", "gc", "--cache-dir", cacheDir, "--yes"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
		t.Fatalf("expected temp file removed, got %v", err)
	}
	if _, err := os.Stat(keepPath); err != nil {
		t.Fatalf("expected normal file kept, got %v", err)
	}
}

func TestModelsInfoShowsRegistryEntry(t *testing.T) {
	cacheDir := t.TempDir()
	err := registry.NewStore(cacheDir).Upsert(registry.Entry{
		RepoID:       "Qwen/Qwen3-0.6B",
		RepoType:     "model",
		Revision:     "main",
		Commit:       "abc123",
		LocalDir:     "./models/qwen3",
		SnapshotPath: cacheDir + "/snapshot",
		FileCount:    2,
		TotalSize:    2048,
		DownloadedAt: time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC),
		Files: []registry.File{
			{Path: "config.json", Size: 512},
			{Path: "model.safetensors", Size: 1536},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"models", "info", "Qwen/Qwen3-0.6B", "--cache-dir", cacheDir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"Repo:        Qwen/Qwen3-0.6B", "Commit:      abc123", "Local path:  ./models/qwen3", "Key files:", "model.safetensors"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected %q in output:\n%s", want, out)
		}
	}
}

func TestModelsInfoJSON(t *testing.T) {
	cacheDir := t.TempDir()
	err := registry.NewStore(cacheDir).Upsert(registry.Entry{
		RepoID:       "Qwen/Qwen3-0.6B",
		RepoType:     "model",
		Revision:     "main",
		Commit:       "abc123",
		SnapshotPath: cacheDir + "/snapshot",
		FileCount:    1,
	})
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"models", "info", "Qwen/Qwen3-0.6B", "--cache-dir", cacheDir, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var entry registry.Entry
	if err := json.Unmarshal(stdout.Bytes(), &entry); err != nil {
		t.Fatal(err)
	}
	if entry.Commit != "abc123" {
		t.Fatalf("unexpected entry: %#v", entry)
	}
}

func TestModelsRemoveRequiresYes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"models", "remove", "Qwen/Qwen3-0.6B", "--cache-dir", t.TempDir()}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "--yes") {
		t.Fatalf("expected --yes warning, got %q", stderr.String())
	}
}

func TestModelsRemoveDeletesLocalFilesAndRegistryOnly(t *testing.T) {
	cacheDir := t.TempDir()
	localDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(localDir, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	nestedDir := filepath.Join(localDir, "sub")
	if err := os.MkdirAll(nestedDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nestedDir, "model.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshotPath := filepath.Join(cacheDir, "snapshot")
	if err := os.MkdirAll(snapshotPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snapshotPath, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := registry.NewStore(cacheDir).Upsert(registry.Entry{
		RepoID:       "Qwen/Qwen3-0.6B",
		RepoType:     "model",
		Revision:     "main",
		Commit:       "abc123",
		LocalDir:     localDir,
		SnapshotPath: snapshotPath,
		FileCount:    2,
		Files: []registry.File{
			{Path: "config.json", Size: 2},
			{Path: "sub/model.safetensors", Size: 7},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"models", "remove", "Qwen/Qwen3-0.6B", "--cache-dir", cacheDir, "--yes"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if _, err := os.Stat(filepath.Join(localDir, "config.json")); !os.IsNotExist(err) {
		t.Fatalf("expected local config to be removed, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(localDir, "sub", "model.safetensors")); !os.IsNotExist(err) {
		t.Fatalf("expected local weights to be removed, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(snapshotPath, "config.json")); err != nil {
		t.Fatalf("expected cache snapshot file to remain: %v", err)
	}
	entries, err := registry.NewStore(cacheDir).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected registry to be empty, got %#v", entries)
	}
}

func TestTokenizeCommand(t *testing.T) {
	dir := t.TempDir()
	writeCLITokenizer(t, dir)

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"tokenize", dir, "hello world!"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if strings.TrimSpace(stdout.String()) != "1 2 3" {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestTokensCommand(t *testing.T) {
	dir := t.TempDir()
	writeCLITokenizer(t, dir)

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"tokens", dir, "hello world!"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if strings.TrimSpace(stdout.String()) != "3" {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestContextCommand(t *testing.T) {
	dir := t.TempDir()
	writeCLITokenizer(t, dir)

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"context", dir, "hello world!", "--max-context", "4"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Tokens:      3") || !strings.Contains(stdout.String(), "Fits:        yes") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"context", dir, "hello world!", "--max-context", "2", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("json code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	var result struct {
		Tokens int  `json:"tokens"`
		Fits   bool `json:"fits"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Tokens != 3 || result.Fits {
		t.Fatalf("unexpected json: %s", stdout.String())
	}
}

func TestContextCommandOverflowExitCode(t *testing.T) {
	dir := t.TempDir()
	writeCLITokenizer(t, dir)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"context", dir, "hello world!", "--max-context", "2"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Fits:        no") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestPromptCommand(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte("<|im_start|>{{ role }}"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"prompt", dir, "--system", "You are helpful", "--user", "Hello"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "<|im_start|>system\nYou are helpful<|im_end|>") || !strings.Contains(out, "<|im_start|>assistant\n") {
		t.Fatalf("unexpected prompt: %q", out)
	}
}

func TestInspectCommandShowsGGUFMetadata(t *testing.T) {
	modelPath := fakeInspectGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"inspect", modelPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"Model type:  llama", "Quantization:  mostly_q4_k_m", "Context:       4096", "Chat template: yes"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected %q in output:\n%s", want, out)
		}
	}
}

func TestInspectCommandJSON(t *testing.T) {
	modelPath := fakeInspectGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"inspect", modelPath, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	var body struct {
		GGUF struct {
			Architecture  string `json:"architecture"`
			ContextLength uint64 `json:"context_length"`
		} `json:"gguf"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.GGUF.Architecture != "llama" || body.GGUF.ContextLength != 4096 {
		t.Fatalf("unexpected json: %s", stdout.String())
	}
}

func TestInspectCommandShowsModelCard(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("---\nlicense: mit\npipeline_tag: text-generation\ntags: [test]\n---\n# Test Model\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"inspect", dir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"Model card:", "Title:        Test Model", "License:      mit", "Pipeline:     text-generation"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected %q in output:\n%s", want, out)
		}
	}
}

func TestInspectCommandShowsSafetensorsMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), cliSafetensorsFixture(t, `{
		"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}
	}`, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"inspect", dir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"Safetensors:", "Tensors:      1", "Parameters:   4", "DTypes:       F16=1"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected %q in output:\n%s", want, out)
		}
	}
}

func TestCheckCommandReportsCompatibility(t *testing.T) {
	modelPath := fakeInspectGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"check", modelPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"Artifact:", "Format:       gguf", "Run:          native", "Train:        native-token-bias", "Backends:", "llama.cpp: yes", "Chat template: yes", "Context:       4096"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected %q in output:\n%s", want, out)
		}
	}
}

func TestCheckCommandJSON(t *testing.T) {
	modelPath := fakeInspectGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"check", modelPath, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	var body struct {
		ChatTemplate bool `json:"chat_template"`
		Artifact     struct {
			Format                string `json:"format"`
			RecommendedRunBackend string `json:"recommended_run_backend"`
		} `json:"artifact"`
		Backends []struct {
			Name       string `json:"name"`
			Compatible bool   `json:"compatible"`
		} `json:"backends"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if !body.ChatTemplate || body.Artifact.Format != "gguf" || body.Artifact.RecommendedRunBackend == "" || len(body.Backends) == 0 {
		t.Fatalf("unexpected check json: %s", stdout.String())
	}
}

func TestRunCommandUsesLlamaBackend(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"run", modelPath, "hello", "--backend", "llama.cpp"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "fake llama output") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestRunCommandPassesRuntimeFlags(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{
		"run",
		modelPath,
		"hello",
		"--backend",
		"llama.cpp",
		"--max-tokens",
		"8",
		"--temperature",
		"0.7",
		"--top-p",
		"0.9",
		"--repeat-penalty",
		"1.2",
		"--seed",
		"42",
		"--stop",
		"END",
		"--threads",
		"4",
		"--ctx-size",
		"2048",
		"--gpu-layers",
		"20",
		"--gpu",
		"full",
		"--main-gpu",
		"1",
		"--tensor-split",
		"3,1",
		"--split-mode",
		"layer",
		"--flash-attn",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	for _, want := range []string{"-n 8", "--temp 0.7", "--top-p 0.9", "--repeat-penalty 1.2", "--seed 42", "--reverse-prompt END", "-t 4", "-c 2048", "-ngl 20", "--main-gpu 1", "--tensor-split 3,1", "--split-mode layer", "-fa"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("expected %q in output: %q", want, stdout.String())
		}
	}
}

func TestRunCommandNativeShortcutUsesNativeBackend(t *testing.T) {
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"run", modelPath, "hello", "--native"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "inspect native GGUF model") {
		t.Fatalf("expected native backend error, got %q", stderr.String())
	}
}

func TestRunCommandRejectsNativeAndBackendTogether(t *testing.T) {
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"run", modelPath, "hello", "--native", "--backend", "llama.cpp"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "either --native or --backend") {
		t.Fatalf("unexpected stderr: %q", stderr.String())
	}
}

func TestRunCommandRejectsInvalidGPUMode(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"run", modelPath, "hello", "--gpu", "fastest"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "gpu must be one of") {
		t.Fatalf("unexpected stderr: %q", stderr.String())
	}
}

func TestRunCommandRejectsInvalidRepeatPenalty(t *testing.T) {
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"run", modelPath, "hello", "--repeat-penalty", "0.5"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "repeat-penalty") {
		t.Fatalf("unexpected stderr: %q", stderr.String())
	}
}

func TestRunCommandAppendsRunLog(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	logPath := filepath.Join(t.TempDir(), "runs.jsonl")
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"run", modelPath, "hello", "--backend", "llama.cpp", "--log", logPath, "--max-tokens", "4"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, `"command":"run"`) || !strings.Contains(text, `"prompt":"hello"`) || !strings.Contains(text, `"max_tokens":4`) {
		t.Fatalf("unexpected log: %s", text)
	}
	if strings.Contains(text, "api_key") {
		t.Fatalf("log should not include api key: %s", text)
	}
}

func TestChatCommandUsesLlamaBackend(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"chat", modelPath, "hello", "--backend", "llama.cpp"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "fake llama output") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestChatCommandNativeShortcutUsesNativeBackend(t *testing.T) {
	modelPath := fakeCLIGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"chat", modelPath, "hello", "--native"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "inspect native GGUF model") {
		t.Fatalf("expected native backend error, got %q", stderr.String())
	}
}

func TestChatCommandSavesSession(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	sessionPath := filepath.Join(t.TempDir(), "chat.json")
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"chat", modelPath, "hello", "--backend", "llama.cpp", "--save", sessionPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	session := readCLIChatSession(t, sessionPath)
	if session.Path != modelPath || len(session.Messages) != 2 {
		t.Fatalf("unexpected session: %#v", session)
	}
	if session.Messages[0].Role != nego.RoleUser || session.Messages[1].Role != nego.RoleAssistant {
		t.Fatalf("unexpected messages: %#v", session.Messages)
	}
}

func TestChatInteractiveStreamsTurns(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	logPath := filepath.Join(t.TempDir(), "runs.jsonl")
	var stdout, stderr bytes.Buffer
	code := RunWithIO(context.Background(), []string{"chat", modelPath, "--backend", "llama.cpp", "--interactive", "--log", logPath}, strings.NewReader("hello\n/exit\n"), &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "Interactive chat") || !strings.Contains(stdout.String(), "assistant> fake llama output") {
		t.Fatalf("unexpected interactive output stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"command":"chat"`) || !strings.Contains(string(data), `"messages"`) {
		t.Fatalf("unexpected log: %s", string(data))
	}
}

func TestChatInteractiveResumesSession(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	sessionPath := filepath.Join(t.TempDir(), "chat.json")
	initial := chatSession{
		Backend: "llama.cpp",
		Path:    modelPath,
		Messages: []nego.Message{
			{Role: nego.RoleUser, Content: "hello"},
			{Role: nego.RoleAssistant, Content: "hi"},
		},
	}
	if err := saveChatSession(sessionPath, initial); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := RunWithIO(context.Background(), []string{"chat", "--session", sessionPath, "--interactive"}, strings.NewReader("again\n/exit\n"), &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	session := readCLIChatSession(t, sessionPath)
	if session.Path != modelPath || len(session.Messages) != 4 {
		t.Fatalf("unexpected resumed session: %#v", session)
	}
	if session.Messages[2].Content != "again" {
		t.Fatalf("unexpected resumed messages: %#v", session.Messages)
	}
}

func TestRunsListAndShow(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	logPath := filepath.Join(t.TempDir(), "runs.jsonl")
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"chat", modelPath, "hello", "--backend", "llama.cpp", "--log", logPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("chat code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"runs", "list", logPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("list code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "COMMAND") || !strings.Contains(out, "chat") {
		t.Fatalf("unexpected list output: %q", out)
	}

	var entries []struct {
		ID string `json:"id"`
	}
	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"runs", "list", logPath, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("json list code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if err := json.Unmarshal(stdout.Bytes(), &entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].ID == "" {
		t.Fatalf("unexpected entries: %s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"runs", "show", logPath, entries[0].ID}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("show code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Messages:") || !strings.Contains(stdout.String(), "Output:") {
		t.Fatalf("unexpected show output: %q", stdout.String())
	}
}

func TestBackendsCommands(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"backends", "list"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("list code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "llama.cpp") || !strings.Contains(out, "native-hf") || !strings.Contains(out, "openai-compatible") {
		t.Fatalf("unexpected list output: %q", out)
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"backends", "info", "llama.cpp"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("info code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "GGUF") || !strings.Contains(stdout.String(), "ctx_size") {
		t.Fatalf("unexpected info output: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"backends", "info", "native"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("native info code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "generate_experimental") || !strings.Contains(stdout.String(), "legacy_quant_dequant") {
		t.Fatalf("unexpected native info output: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"backends", "info", "openai-compatible", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("json info code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	var info struct {
		Name         string   `json:"name"`
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &info); err != nil {
		t.Fatal(err)
	}
	if info.Name != "openai-compatible" || len(info.Capabilities) == 0 {
		t.Fatalf("unexpected backend info: %s", stdout.String())
	}
}

func TestDatasetCommands(t *testing.T) {
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "data.jsonl")
	body := strings.Join([]string{
		`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}`,
		`{"messages":[{"role":"user","content":"bye"},{"role":"assistant","content":"later"}]}`,
		`{"messages":[{"role":"user","content":"ok"},{"role":"assistant","content":"yes"}]}`,
	}, "\n") + "\n"
	if err := os.WriteFile(dataPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"dataset", "inspect", dataPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("inspect code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Rows:        3") || !strings.Contains(stdout.String(), "chat: 3") {
		t.Fatalf("unexpected inspect output: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"dataset", "validate", dataPath, "--format", "chat"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("validate code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}

	csvPath := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvPath, []byte("prompt,completion,split\nhi,hello,train\nbye,later,test\n,missing,train\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	convertedOut := filepath.Join(dir, "converted.jsonl")
	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"dataset", "convert", csvPath, "--out", convertedOut, "--select", "prompt,completion", "--require", "prompt,completion"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("convert code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	convertedData, err := os.ReadFile(convertedOut)
	if err != nil {
		t.Fatal(err)
	}
	converted := string(convertedData)
	if strings.Count(strings.TrimSpace(converted), "\n")+1 != 2 || strings.Contains(converted, "split") {
		t.Fatalf("unexpected converted output: %q", converted)
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"dataset", "filter", csvPath, "--where", "split=train", "--where", "prompt", "--select", "prompt", "--out", "-"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("filter code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if strings.Count(strings.TrimSpace(stdout.String()), "\n")+1 != 1 || !strings.Contains(stdout.String(), `"prompt":"hi"`) {
		t.Fatalf("unexpected filter output: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"dataset", "sample", dataPath, "--n", "2"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("sample code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if got := strings.Count(strings.TrimSpace(stdout.String()), "\n") + 1; got != 2 {
		t.Fatalf("sample rows = %d output=%q", got, stdout.String())
	}

	trainOut := filepath.Join(dir, "train.jsonl")
	testOut := filepath.Join(dir, "test.jsonl")
	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"dataset", "split", dataPath, "--train-out", trainOut, "--test-out", testOut, "--test-size", "0.34"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("split code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if _, err := os.Stat(trainOut); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(testOut); err != nil {
		t.Fatal(err)
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"dataset", "split", dataPath, "--train-out", trainOut, "--test-out", filepath.Join(dir, "test2.jsonl")}, &stdout, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "refusing to overwrite") {
		t.Fatalf("expected overwrite refusal, code=%d stderr=%q", code, stderr.String())
	}
}

func TestRuntimeOptionsMergesConfigAndFlags(t *testing.T) {
	got := runtimeOptions(map[string]string{
		"threads":    "2",
		"ctx_size":   "1024",
		"gpu_layers": "8",
		"gpu":        "off",
		"custom":     "value",
	}, runtimeFlagOptions{
		threads:        4,
		ctxSize:        2048,
		gpuMode:        "full",
		mainGPU:        1,
		tensorSplit:    "3,1",
		splitMode:      "layer",
		flashAttention: true,
		adapterPath:    "adapter.json",
	})
	want := map[string]string{
		"threads":      "4",
		"ctx_size":     "2048",
		"gpu_layers":   "8",
		"gpu":          "full",
		"main_gpu":     "1",
		"tensor_split": "3,1",
		"split_mode":   "layer",
		"flash_attn":   "true",
		"adapter_path": "adapter.json",
		"custom":       "value",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("runtimeOptions = %#v, want %#v", got, want)
	}
}

func TestValidateRuntimeOptions(t *testing.T) {
	if err := validateRuntimeOptions(map[string]string{"gpu": "full", "split_mode": "row", "gpu_layers": "32"}); err != nil {
		t.Fatal(err)
	}
	if err := validateRuntimeOptions(map[string]string{"gpu": "fastest"}); err == nil {
		t.Fatal("expected invalid gpu mode error")
	}
	if err := validateRuntimeOptions(map[string]string{"gpu_layers": "many"}); err == nil {
		t.Fatal("expected invalid gpu_layers error")
	}
	if err := validateRuntimeOptions(map[string]string{"main_gpu": "-1"}); err == nil {
		t.Fatal("expected invalid main_gpu error")
	}
	if err := validateRuntimeOptions(map[string]string{"flash_attn": "maybe"}); err == nil {
		t.Fatal("expected invalid flash_attn error")
	}
}

func TestRuntimeLogEntrySanitizesEndpoint(t *testing.T) {
	entry := runtimeLogEntry("run", "openai-compatible", "", "https://user:secret@example.com/v1?api_key=secret", "model", "hello", nil, "world", time.Now(), 0, 0, 0, 0, nil, 0, nil, nil)
	if entry.Endpoint != "https://example.com/v1" {
		t.Fatalf("Endpoint = %q", entry.Endpoint)
	}
}

func TestChatSessionSanitizesEndpoint(t *testing.T) {
	session := chatSession{}.withMessages("openai-compatible", "", "https://user:secret@example.com/v1?api_key=secret", "model", nil)
	if session.Endpoint != "https://example.com/v1" {
		t.Fatalf("Endpoint = %q", session.Endpoint)
	}
}

func TestRunCommandUsesConfigFile(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	configPath := filepath.Join(t.TempDir(), "nego.json")
	body := `{"backend":"llama.cpp","path":` + strconv.Quote(modelPath) + `,"prompt":"hello","max_tokens":4}`
	if err := os.WriteFile(configPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"run", "-f", configPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "fake llama output") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestChatCommandUsesConfigFile(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	modelPath := fakeCLIGGUF(t)
	configPath := filepath.Join(t.TempDir(), "nego.json")
	body := `{"backend":"llama.cpp","path":` + strconv.Quote(modelPath) + `,"system":"Helpful","prompt":"hello"}`
	if err := os.WriteFile(configPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"chat", "-f", configPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "fake llama output") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestEmbedCommand(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1, 0}}},
		})
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"embed", "hello world", "--endpoint", server.URL, "--model", "embed-model"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), `"embeddings":[[1,0]]`) {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestEvalCommand(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"text": "generated: hello"}},
		})
	}))
	defer server.Close()

	suitePath := filepath.Join(t.TempDir(), "suite.json")
	body := `{
		"backend":"openai-compatible",
		"endpoint":` + strconv.Quote(server.URL) + `,
		"model":"test-model",
		"cases":[{"name":"hello","prompt":"hello","contains":"generated"}]
	}`
	if err := os.WriteFile(suitePath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"eval", suitePath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Passed: 1") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestEvalReportAndCompareCommands(t *testing.T) {
	dir := t.TempDir()
	baselinePath := filepath.Join(dir, "baseline.json")
	candidatePath := filepath.Join(dir, "candidate.json")
	baseline := `{"passed":1,"failed":1,"results":[{"name":"fixed","passed":false,"error":"bad"},{"name":"regressed","passed":true}]}`
	candidate := `{"passed":1,"failed":1,"results":[{"name":"fixed","passed":true},{"name":"regressed","passed":false,"error":"worse"}]}`
	if err := os.WriteFile(baselinePath, []byte(baseline), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(candidatePath, []byte(candidate), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"eval", "report", candidatePath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("report code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "PassRate: 50.00%") || !strings.Contains(stdout.String(), "regressed") {
		t.Fatalf("unexpected report output: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"eval", "compare", baselinePath, candidatePath}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("compare code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Improvements:") || !strings.Contains(stdout.String(), "Regressions:") {
		t.Fatalf("unexpected compare output: %q", stdout.String())
	}
}

func TestVersionCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"version"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "nego dev") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestConvertGGUFCommand(t *testing.T) {
	out := filepath.Join(t.TempDir(), "model.gguf")
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"convert", "gguf", "model-dir", "--out", out, "--converter", fakeConverterCommand(t)}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatal(err)
	}
}

func TestTrainCommand(t *testing.T) {
	jobPath := filepath.Join(t.TempDir(), "job.json")
	body := `{"name":"test","command":` + strconv.Quote(fakeTrainingCommand(t)) + `,"args":["hello"]}`
	if err := os.WriteFile(jobPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"train", jobPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "training hello") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestTrainValidateCommand(t *testing.T) {
	jobPath := filepath.Join(t.TempDir(), "job.json")
	body := `{"name":"test","command":` + strconv.Quote(fakeTrainingCommand(t)) + `,"args":["hello"]}`
	if err := os.WriteFile(jobPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"train", "validate", jobPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Training job is valid") {
		t.Fatalf("unexpected stdout: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"train", "validate", jobPath, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("json code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), `"valid":true`) {
		t.Fatalf("unexpected json: %q", stdout.String())
	}
}

func TestTrainCheckCommand(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	jobPath := filepath.Join(dir, "job.json")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	body := `{"name":"test","base_model":` + strconv.Quote(modelDir) + `,"train_file":` + strconv.Quote(trainFile) + `,"dataset_format":"completion","command":` + strconv.Quote(fakeTrainingCommand(t)) + `}`
	if err := os.WriteFile(jobPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"train", "check", jobPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Training job check passed") || !strings.Contains(stdout.String(), "Train file:") || !strings.Contains(stdout.String(), "(1 rows)") {
		t.Fatalf("unexpected stdout: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = Run(context.Background(), []string{"train", "check", jobPath, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("json code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), `"valid":true`) || !strings.Contains(stdout.String(), `"train_rows":1`) {
		t.Fatalf("unexpected json: %q", stdout.String())
	}
}

func TestTrainCapabilitiesCommand(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"num_hidden_layers":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), cliSafetensorsFixture(t, `{
		"model.embed_tokens.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}
	}`, 4), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"train", "capabilities", modelDir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Training capabilities") || !strings.Contains(out, "token-bias") || !strings.Contains(out, "native-lora") {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestTrainNativeCommandCreatesAdapter(t *testing.T) {
	modelPath := fakeInspectGGUF(t)
	dir := t.TempDir()
	trainFile := filepath.Join(dir, "train.jsonl")
	outputDir := filepath.Join(dir, "adapter")
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{
		"train",
		"native",
		modelPath,
		"--train-file",
		trainFile,
		"--dataset-format",
		"completion",
		"--out",
		outputDir,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Native training completed") || !strings.Contains(stdout.String(), "Adapter:") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
	if _, err := os.Stat(filepath.Join(outputDir, "adapter.json")); err != nil {
		t.Fatal(err)
	}
}

func TestTrainCheckCommandRedactsSensitiveArgs(t *testing.T) {
	dir := t.TempDir()
	jobPath := filepath.Join(dir, "job.json")
	body := `{"name":"secret","command":` + strconv.Quote(fakeTrainingCommand(t)) + `,"args":["--token","super-secret","API_KEY=also-secret"]}`
	if err := os.WriteFile(jobPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"train", "check", jobPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if strings.Contains(stdout.String(), "super-secret") || strings.Contains(stdout.String(), "also-secret") {
		t.Fatalf("sensitive value leaked: %q", stdout.String())
	}
	if !strings.Contains(stdout.String(), "<redacted>") {
		t.Fatalf("expected redaction: %q", stdout.String())
	}
}

func TestTrainValidateCommandReportsInvalidJob(t *testing.T) {
	jobPath := filepath.Join(t.TempDir(), "job.json")
	if err := os.WriteFile(jobPath, []byte(`{"name":"bad"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"train", "validate", jobPath}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "training command is required") {
		t.Fatalf("unexpected stderr: %q", stderr.String())
	}
}

func TestTrainInitCommand(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", "qwen3")
	trainFile := filepath.Join(dir, "data", "train.jsonl")
	evalFile := filepath.Join(dir, "data", "test.jsonl")
	jobPath := filepath.Join(dir, "train-job.json")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(trainFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(evalFile, []byte(`{"prompt":"bye","completion":"later"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{
		"train",
		"init",
		"--name",
		"qwen3-lora",
		"--base-model",
		modelDir,
		"--train-file",
		trainFile,
		"--eval-file",
		evalFile,
		"--output-dir",
		filepath.Join(dir, "outputs", "qwen3-lora"),
		"--out",
		jobPath,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	data, err := os.ReadFile(jobPath)
	if err != nil {
		t.Fatal(err)
	}
	var spec training.JobSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		t.Fatal(err)
	}
	if spec.BaseModel != modelDir || spec.TrainFile != trainFile || spec.OutputDir == "" {
		t.Fatalf("unexpected spec: %#v", spec)
	}
	if spec.DatasetFormat != "auto" {
		t.Fatalf("unexpected dataset format: %#v", spec)
	}
	if !strings.Contains(stdout.String(), "Training job:") {
		t.Fatalf("unexpected stdout: %q", stdout.String())
	}
}

func TestShareManifestCommand(t *testing.T) {
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
	outPath := filepath.Join(dir, "share-manifest.json")
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"share", "manifest", modelDir, "--out", outPath, "--repo", "user/model", "--base-model", "Qwen/Qwen3-0.6B"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Share manifest:") {
		t.Fatalf("unexpected stdout: %q", stdout.String())
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"repo_id": "user/model"`) || !strings.Contains(string(data), `"adapter.safetensors"`) {
		t.Fatalf("unexpected manifest: %s", string(data))
	}
}

func TestSharePackageCommand(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "model")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "README.md"), []byte("model card"), 0o644); err != nil {
		t.Fatal(err)
	}
	outPath := filepath.Join(dir, "model.tar.gz")
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"share", "package", modelDir, "--out", outPath, "--repo", "user/model"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Share package:") {
		t.Fatalf("unexpected stdout: %q", stdout.String())
	}
	if info, err := os.Stat(outPath); err != nil || info.Size() == 0 {
		t.Fatalf("archive was not written: info=%#v err=%v", info, err)
	}
}

func TestShareUploadCommand(t *testing.T) {
	var sawUpload bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/models/user/model/commit/main" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.String())
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Fatalf("Authorization = %q", got)
		}
		sawUpload = true
		_, _ = w.Write([]byte(`{"oid":"abc123","commitUrl":"https://huggingface.co/user/model/commit/abc123"}`))
	}))
	defer server.Close()

	filePath := filepath.Join(t.TempDir(), "README.md")
	if err := os.WriteFile(filePath, []byte("model card"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"share", "upload", "user/model", filePath, "README.md", "--endpoint", server.URL, "--token", "test-token"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !sawUpload || !strings.Contains(stdout.String(), "Uploaded:") || !strings.Contains(stdout.String(), "abc123") {
		t.Fatalf("unexpected upload output: saw=%v stdout=%q", sawUpload, stdout.String())
	}
}

func writeCLITokenizer(t *testing.T, dir string) {
	t.Helper()
	body := `{
		"model": {
			"type": "WordLevel",
			"unk_token": "[UNK]",
			"vocab": {
				"[UNK]": 0,
				"hello": 1,
				"Ġworld": 2,
				"!": 3
			}
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fakeCLILlamaCommand(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, "fake-llama.bat")
		if err := os.WriteFile(path, []byte("@echo off\necho fake llama output %*\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	path := filepath.Join(dir, "fake-llama.sh")
	if err := os.WriteFile(path, []byte("#!/bin/sh\necho fake llama output \"$@\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func readCLIChatSession(t *testing.T, path string) chatSession {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var session chatSession
	if err := json.Unmarshal(data, &session); err != nil {
		t.Fatal(err)
	}
	return session
}

func fakeCLIGGUF(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func cliSafetensorsFixture(t *testing.T, header string, dataBytes int) []byte {
	t.Helper()
	header = strings.TrimSpace(header)
	data := make([]byte, 8+len(header)+dataBytes)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	return data
}

func fakeInspectGGUF(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	for _, value := range []any{uint32(3), uint64(2), uint64(5)} {
		if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeInspectGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeInspectGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeInspectGGUFUint32KV(t, &buf, "llama.context_length", 4096)
	writeInspectGGUFStringKV(t, &buf, "tokenizer.chat_template", "[INST] {{ message }} [/INST]")
	writeInspectGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeInspectGGUFTensor(t, &buf, "token_embd.weight", []uint64{2, 16}, 1, 0)
	writeInspectGGUFTensor(t, &buf, "blk.0.attn_q.weight", []uint64{16, 16}, 12, 128)
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeInspectGGUFStringKV(t *testing.T, buf *bytes.Buffer, key, value string) {
	t.Helper()
	writeInspectGGUFString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(8)); err != nil {
		t.Fatal(err)
	}
	writeInspectGGUFString(t, buf, value)
}

func writeInspectGGUFUint32KV(t *testing.T, buf *bytes.Buffer, key string, value uint32) {
	t.Helper()
	writeInspectGGUFString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(4)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
		t.Fatal(err)
	}
}

func writeInspectGGUFStringArrayKV(t *testing.T, buf *bytes.Buffer, key string, values []string) {
	t.Helper()
	writeInspectGGUFString(t, buf, key)
	for _, value := range []any{uint32(9), uint32(8), uint64(len(values))} {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range values {
		writeInspectGGUFString(t, buf, value)
	}
}

func writeInspectGGUFTensor(t *testing.T, buf *bytes.Buffer, name string, shape []uint64, typ uint32, offset uint64) {
	t.Helper()
	writeInspectGGUFString(t, buf, name)
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(shape))); err != nil {
		t.Fatal(err)
	}
	for _, dim := range shape {
		if err := binary.Write(buf, binary.LittleEndian, dim); err != nil {
			t.Fatal(err)
		}
	}
	if err := binary.Write(buf, binary.LittleEndian, typ); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(buf, binary.LittleEndian, offset); err != nil {
		t.Fatal(err)
	}
}

func writeInspectGGUFString(t *testing.T, buf *bytes.Buffer, value string) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	if _, err := buf.WriteString(value); err != nil {
		t.Fatal(err)
	}
}

func fakeConverterCommand(t *testing.T) string {
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

func fakeTrainingCommand(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, "train.bat")
		if err := os.WriteFile(path, []byte("@echo off\necho training %*\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	path := filepath.Join(dir, "train.sh")
	if err := os.WriteFile(path, []byte("#!/bin/sh\necho training \"$@\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}
