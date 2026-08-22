package cli

import (
	"bytes"
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

	"github.com/gakon/nego-ai/hub"
	"github.com/gakon/nego-ai/internal/registry"
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
	code := Run(t.Context(), []string{"models", "list", "--cache-dir", cacheDir}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"models", "list", "--cache-dir", cacheDir, "--json"}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"models", "list", "--cache-dir", t.TempDir()}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"cache", "usage", "--cache-dir", cacheDir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Registered models: 1") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestCacheGCRequiresYes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(t.Context(), []string{"cache", "gc", "--cache-dir", t.TempDir()}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"cache", "gc", "--cache-dir", cacheDir, "--yes"}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"models", "info", "Qwen/Qwen3-0.6B", "--cache-dir", cacheDir}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"models", "info", "Qwen/Qwen3-0.6B", "--cache-dir", cacheDir, "--json"}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"models", "remove", "Qwen/Qwen3-0.6B", "--cache-dir", t.TempDir()}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"models", "remove", "Qwen/Qwen3-0.6B", "--cache-dir", cacheDir, "--yes"}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"tokenize", dir, "hello world!"}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"tokens", dir, "hello world!"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if strings.TrimSpace(stdout.String()) != "3" {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestPromptCommand(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte("<|im_start|>{{ role }}"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(t.Context(), []string{"prompt", dir, "--system", "You are helpful", "--user", "Hello"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "<|im_start|>system\nYou are helpful<|im_end|>") || !strings.Contains(out, "<|im_start|>assistant\n") {
		t.Fatalf("unexpected prompt: %q", out)
	}
}

func TestRunCommandUsesLlamaBackend(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	var stdout, stderr bytes.Buffer
	code := Run(t.Context(), []string{"run", "model.gguf", "hello"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "fake llama output") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestChatCommandUsesLlamaBackend(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	var stdout, stderr bytes.Buffer
	code := Run(t.Context(), []string{"chat", "model.gguf", "hello"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "fake llama output") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestRunCommandUsesConfigFile(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	configPath := filepath.Join(t.TempDir(), "nego.json")
	if err := os.WriteFile(configPath, []byte(`{"path":"model.gguf","prompt":"hello","max_tokens":4}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(t.Context(), []string{"run", "-f", configPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "fake llama output") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestChatCommandUsesConfigFile(t *testing.T) {
	t.Setenv("NEGO_LLAMA_CLI", fakeCLILlamaCommand(t))
	configPath := filepath.Join(t.TempDir(), "nego.json")
	if err := os.WriteFile(configPath, []byte(`{"path":"model.gguf","system":"Helpful","prompt":"hello"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := Run(t.Context(), []string{"chat", "-f", configPath}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"embed", "hello world", "--endpoint", server.URL, "--model", "embed-model"}, &stdout, &stderr)
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
	code := Run(t.Context(), []string{"eval", suitePath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "Passed: 1") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}

func TestVersionCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(t.Context(), []string{"version"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "nego dev") {
		t.Fatalf("unexpected output: %q", stdout.String())
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
