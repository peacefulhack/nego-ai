package runs

import (
	"path/filepath"
	"testing"
	"time"
)

func TestAppendReadAndFind(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runs.jsonl")
	entry := Entry{
		ID:         "run-1",
		Command:    "run",
		Backend:    "llama.cpp",
		Path:       "model.gguf",
		Prompt:     "hello",
		Output:     "world",
		StartedAt:  time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC),
		DurationMS: 12,
		MaxTokens:  4,
	}
	if err := Append(path, entry); err != nil {
		t.Fatal(err)
	}
	entries, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].ID != "run-1" || entries[0].Output != "world" {
		t.Fatalf("unexpected entries: %#v", entries)
	}
	got, ok := Find(entries, "run-1")
	if !ok || got.Command != "run" {
		t.Fatalf("Find returned %#v, %v", got, ok)
	}
}

func TestAppendGeneratesIDAndTimestamp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runs.jsonl")
	if err := Append(path, Entry{Command: "run"}); err != nil {
		t.Fatal(err)
	}
	entries, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].ID == "" || entries[0].StartedAt.IsZero() {
		t.Fatalf("unexpected entry: %#v", entries)
	}
}
