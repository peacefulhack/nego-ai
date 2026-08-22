package training

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRunTrainingJob(t *testing.T) {
	result, err := Run(context.Background(), JobSpec{
		Name:    "test",
		Command: fakeTrainingCommand(t),
		Args:    []string{"hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Success || !strings.Contains(result.Stdout, "training hello") {
		t.Fatalf("result = %#v", result)
	}
}

func TestRunRequiresCommand(t *testing.T) {
	if _, err := Run(context.Background(), JobSpec{}); err == nil {
		t.Fatal("expected validation error")
	}
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
