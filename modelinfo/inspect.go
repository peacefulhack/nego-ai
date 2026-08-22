package modelinfo

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
)

type Info struct {
	Path             string         `json:"path"`
	ModelType        string         `json:"model_type,omitempty"`
	Architectures    []string       `json:"architectures,omitempty"`
	Config           map[string]any `json:"config,omitempty"`
	GenerationConfig map[string]any `json:"generation_config,omitempty"`
	GGUF             *GGUFInfo      `json:"gguf,omitempty"`
	Files            []File         `json:"files,omitempty"`
}

type File struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	Kind string `json:"kind"`
}

func Inspect(path string) (*Info, error) {
	root, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	info := &Info{Path: root}
	stat, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !stat.IsDir() {
		kind := fileKind(root)
		if kind != "" {
			info.Files = []File{{Path: filepath.Base(root), Size: stat.Size(), Kind: kind}}
		}
		if kind == "gguf" {
			gguf, err := InspectGGUF(root)
			if err != nil {
				return nil, err
			}
			info.GGUF = gguf
			applyGGUFModelFields(info, gguf)
		}
		return info, nil
	}

	config, err := readJSON(filepath.Join(root, "config.json"))
	if err != nil {
		return nil, err
	}
	info.Config = config
	info.ModelType = stringValue(config["model_type"])
	info.Architectures = stringSlice(config["architectures"])

	generationConfig, err := readJSON(filepath.Join(root, "generation_config.json"))
	if err != nil {
		return nil, err
	}
	info.GenerationConfig = generationConfig

	files, err := inspectFiles(root)
	if err != nil {
		return nil, err
	}
	info.Files = files
	if path := firstFileKind(root, files, "gguf"); path != "" {
		gguf, err := InspectGGUF(path)
		if err != nil {
			return nil, err
		}
		info.GGUF = gguf
		applyGGUFModelFields(info, gguf)
	}
	return info, nil
}

func readJSON(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func inspectFiles(root string) ([]File, error) {
	var files []File
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		kind := fileKind(rel)
		if kind == "" {
			return nil
		}
		fileInfo, err := d.Info()
		if err != nil {
			return err
		}
		files = append(files, File{Path: rel, Size: fileInfo.Size(), Kind: kind})
		return nil
	})
	return files, err
}

func fileKind(path string) string {
	lower := strings.ToLower(path)
	switch lower {
	case "config.json":
		return "config"
	case "generation_config.json":
		return "generation_config"
	case "tokenizer.json":
		return "tokenizer"
	case "tokenizer_config.json":
		return "tokenizer_config"
	case "chat_template.jinja":
		return "chat_template"
	}
	switch {
	case strings.HasSuffix(lower, ".safetensors"):
		return "safetensors"
	case strings.HasSuffix(lower, ".gguf"):
		return "gguf"
	case strings.HasSuffix(lower, ".onnx"):
		return "onnx"
	default:
		return ""
	}
}

func stringValue(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}

func stringSlice(value any) []string {
	values, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if s, ok := value.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func firstFileKind(root string, files []File, kind string) string {
	for _, file := range files {
		if file.Kind == kind {
			return filepath.Join(root, filepath.FromSlash(file.Path))
		}
	}
	return ""
}

func applyGGUFModelFields(info *Info, gguf *GGUFInfo) {
	if info.ModelType == "" {
		info.ModelType = gguf.Architecture
	}
	if len(info.Architectures) == 0 && gguf.Architecture != "" {
		info.Architectures = []string{gguf.Architecture}
	}
}
