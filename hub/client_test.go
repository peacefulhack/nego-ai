package hub

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/gakon/nego-ai/internal/registry"
)

func TestDownloadFile(t *testing.T) {
	const body = `{"bos_token":"<s>"}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead && r.URL.Path == "/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json":
			w.Header().Set("X-Repo-Commit", "abc123")
			w.Header().Set("ETag", "tokenizer-etag")
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		case r.Method == http.MethodGet && r.URL.Path == "/Qwen/Qwen3-0.6B/resolve/abc123/tokenizer.json":
			w.Header().Set("X-Repo-Commit", "abc123")
			w.Header().Set("ETag", "tokenizer-etag")
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			_, _ = w.Write([]byte(body))
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.String())
		}
	}))
	defer server.Close()

	localDir := t.TempDir()
	cacheDir := t.TempDir()
	client := NewClient(WithEndpoint(server.URL))
	got, err := client.DownloadFile(context.Background(), DownloadFileOptions{
		RepoID:   "Qwen/Qwen3-0.6B",
		Filename: "tokenizer.json",
		LocalDir: localDir,
		CacheDir: cacheDir,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != filepath.Join(localDir, "tokenizer.json") {
		t.Fatalf("unexpected path: %s", got)
	}
	data, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != body {
		t.Fatalf("unexpected file content: %q", data)
	}
	entries, err := registry.NewStore(cacheDir).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected one registry entry, got %#v", entries)
	}
	if entries[0].RepoID != "Qwen/Qwen3-0.6B" || entries[0].Commit != "abc123" || entries[0].FileCount != 1 {
		t.Fatalf("unexpected registry entry: %#v", entries[0])
	}
}

func TestResolveGGUFFileChoosesPreferredQuantFromSameOwnerRepo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/Qwen/Qwen3-0.6B-GGUF/tree/main":
			_, _ = w.Write([]byte(`[
				{"type":"file","path":"Qwen3-0.6B-Q8_0.gguf","size":805},
				{"type":"file","path":"Qwen3-0.6B-Q4_K_M.gguf","size":484},
				{"type":"file","path":"README.md","size":12}
			]`))
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.String())
		}
	}))
	defer server.Close()

	client := NewClient(WithEndpoint(server.URL))
	got, err := client.ResolveGGUFFile(context.Background(), ResolveGGUFFileOptions{
		RepoID: "Qwen/Qwen3-0.6B",
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.RepoID != "Qwen/Qwen3-0.6B-GGUF" || got.Filename != "Qwen3-0.6B-Q4_K_M.gguf" {
		t.Fatalf("unexpected GGUF selection: %#v", got)
	}
}

func TestResolveGGUFFileFallsBackToOriginalRepo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/acme/tiny-GGUF/tree/main":
			http.NotFound(w, r)
		case "/api/models/acme/tiny/tree/main":
			_, _ = w.Write([]byte(`[{"type":"file","path":"tiny-q4_k_m.gguf","size":128}]`))
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.String())
		}
	}))
	defer server.Close()

	client := NewClient(WithEndpoint(server.URL))
	got, err := client.ResolveGGUFFile(context.Background(), ResolveGGUFFileOptions{RepoID: "acme/tiny"})
	if err != nil {
		t.Fatal(err)
	}
	if got.RepoID != "acme/tiny" || got.Filename != "tiny-q4_k_m.gguf" {
		t.Fatalf("unexpected GGUF selection: %#v", got)
	}
}

func TestResolveGGUFFileUsesOverrideAndFilename(t *testing.T) {
	got, err := NewClient().ResolveGGUFFile(context.Background(), ResolveGGUFFileOptions{
		RepoID:   "Qwen/Qwen3-0.6B",
		GGUFRepo: "unsloth/Qwen3-0.6B-GGUF",
		Filename: "Qwen3-0.6B-Q4_K_M.gguf",
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.RepoID != "unsloth/Qwen3-0.6B-GGUF" || got.Filename != "Qwen3-0.6B-Q4_K_M.gguf" {
		t.Fatalf("unexpected GGUF selection: %#v", got)
	}
}

func TestResolveGGUFFileDoesNotGuessThirdPartyRepos(t *testing.T) {
	got := ggufRepoCandidates("Qwen/Qwen3-0.6B", "")
	joined := strings.Join(got, ",")
	if strings.Contains(joined, "unsloth/") || strings.Contains(joined, "bartowski/") {
		t.Fatalf("unexpected third-party GGUF candidates: %#v", got)
	}
}

func TestDownloadSnapshotWithFilters(t *testing.T) {
	files := map[string]string{
		"tokenizer.json":     `{"tokenizer":true}`,
		"model.safetensors":  "weights",
		"flax_model.msgpack": "ignored",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/Qwen/Qwen3-0.6B/revision/main":
			_, _ = w.Write([]byte(`{"sha":"abc123"}`))
		case "/api/models/Qwen/Qwen3-0.6B/tree/main":
			_, _ = w.Write([]byte(`[
				{"type":"file","path":"tokenizer.json","size":18,"blob_id":"tok"},
				{"type":"file","path":"model.safetensors","size":7,"blob_id":"weights"},
				{"type":"file","path":"flax_model.msgpack","size":7,"blob_id":"msgpack"}
			]`))
		case "/Qwen/Qwen3-0.6B/resolve/abc123/tokenizer.json",
			"/Qwen/Qwen3-0.6B/resolve/abc123/model.safetensors",
			"/Qwen/Qwen3-0.6B/resolve/abc123/flax_model.msgpack":
			name := path.Base(r.URL.Path)
			body := files[name]
			w.Header().Set("X-Repo-Commit", "abc123")
			w.Header().Set("ETag", name)
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			_, _ = w.Write([]byte(body))
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.String())
		}
	}))
	defer server.Close()

	localDir := t.TempDir()
	cacheDir := t.TempDir()
	client := NewClient(WithEndpoint(server.URL))
	got, err := client.DownloadSnapshot(context.Background(), DownloadSnapshotOptions{
		RepoID:   "Qwen/Qwen3-0.6B",
		LocalDir: localDir,
		CacheDir: cacheDir,
		Include:  []string{"*.json", "*.safetensors"},
		Exclude:  []string{"*.msgpack"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != localDir {
		t.Fatalf("unexpected path: %s", got)
	}
	if _, err := os.Stat(filepath.Join(localDir, "tokenizer.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(localDir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(localDir, "flax_model.msgpack")); !os.IsNotExist(err) {
		t.Fatalf("excluded file exists or stat failed differently: %v", err)
	}
	entries, err := registry.NewStore(cacheDir).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected one registry entry, got %#v", entries)
	}
	if entries[0].FileCount != 2 || entries[0].TotalSize != 25 {
		t.Fatalf("unexpected registry sizes: %#v", entries[0])
	}
}
