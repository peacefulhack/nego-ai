package cli

import (
	"bytes"
	"encoding/json"
	"reflect"
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
