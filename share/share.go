package share

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type Manifest struct {
	Path      string    `json:"path"`
	RepoID    string    `json:"repo_id,omitempty"`
	BaseModel string    `json:"base_model,omitempty"`
	Files     []File    `json:"files"`
	TotalSize int64     `json:"total_size"`
	CreatedAt time.Time `json:"created_at"`
}

type File struct {
	Path   string `json:"path"`
	Kind   string `json:"kind,omitempty"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type ManifestOptions struct {
	Path      string
	RepoID    string
	BaseModel string
	Exclude   []string
}

func BuildManifest(opts ManifestOptions) (*Manifest, error) {
	if opts.Path == "" {
		return nil, fmt.Errorf("model path is required")
	}
	root, err := filepath.Abs(opts.Path)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("model path %q is not available: %w", opts.Path, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("model path %q must be a directory", opts.Path)
	}
	manifest := &Manifest{
		Path:      root,
		RepoID:    opts.RepoID,
		BaseModel: opts.BaseModel,
		CreatedAt: time.Now().UTC(),
	}
	excluded, err := excludedPaths(opts.Exclude)
	if err != nil {
		return nil, err
	}
	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if excluded[path] {
			return nil
		}
		file, err := manifestFile(root, path, entry)
		if err != nil {
			return err
		}
		manifest.Files = append(manifest.Files, file)
		manifest.TotalSize += file.Size
		return nil
	}); err != nil {
		return nil, err
	}
	sort.Slice(manifest.Files, func(i, j int) bool {
		return manifest.Files[i].Path < manifest.Files[j].Path
	})
	return manifest, nil
}

func excludedPaths(paths []string) (map[string]bool, error) {
	out := make(map[string]bool, len(paths))
	for _, path := range paths {
		if path == "" {
			continue
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return nil, err
		}
		out[abs] = true
	}
	return out, nil
}

func WriteManifest(path string, manifest *Manifest) error {
	if path == "" {
		return fmt.Errorf("manifest output path is required")
	}
	if manifest == nil {
		return fmt.Errorf("manifest is nil")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func manifestFile(root, path string, entry os.DirEntry) (File, error) {
	info, err := entry.Info()
	if err != nil {
		return File{}, err
	}
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return File{}, err
	}
	sum, err := fileSHA256(path)
	if err != nil {
		return File{}, err
	}
	return File{
		Path:   filepath.ToSlash(rel),
		Kind:   fileKind(path),
		Size:   info.Size(),
		SHA256: sum,
	}, nil
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func fileKind(path string) string {
	name := strings.ToLower(filepath.Base(path))
	switch {
	case name == "readme.md":
		return "model_card"
	case name == "config.json":
		return "config"
	case name == "tokenizer.json" || name == "tokenizer_config.json":
		return "tokenizer"
	case strings.HasSuffix(name, ".safetensors"):
		return "safetensors"
	case strings.HasSuffix(name, ".gguf"):
		return "gguf"
	case strings.HasSuffix(name, ".onnx"):
		return "onnx"
	case strings.HasSuffix(name, ".jsonl"):
		return "dataset"
	default:
		return ""
	}
}
