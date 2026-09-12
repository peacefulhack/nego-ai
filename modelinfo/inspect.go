package modelinfo

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const maxModelCardBytes = 2 << 20

type Info struct {
	Path             string             `json:"path"`
	ModelType        string             `json:"model_type,omitempty"`
	Architectures    []string           `json:"architectures,omitempty"`
	Config           map[string]any     `json:"config,omitempty"`
	GenerationConfig map[string]any     `json:"generation_config,omitempty"`
	Generation       *GenerationConfig  `json:"generation,omitempty"`
	Card             *Card              `json:"card,omitempty"`
	GGUF             *GGUFInfo          `json:"gguf,omitempty"`
	Safetensors      *SafetensorsInfo   `json:"safetensors,omitempty"`
	HFSpec           *HFModelSpec       `json:"hf_spec,omitempty"`
	HFWeights        *HFWeightManifest  `json:"hf_weights,omitempty"`
	HFShapes         *HFShapeReport     `json:"hf_shapes,omitempty"`
	NativeAdapter    *NativeAdapterInfo `json:"native_adapter,omitempty"`
	Files            []File             `json:"files,omitempty"`
}

type Card struct {
	Title       string   `json:"title,omitempty"`
	License     string   `json:"license,omitempty"`
	PipelineTag string   `json:"pipeline_tag,omitempty"`
	LibraryName string   `json:"library_name,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Languages   []string `json:"languages,omitempty"`
	Datasets    []string `json:"datasets,omitempty"`
	BaseModels  []string `json:"base_models,omitempty"`
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
		if kind == "model_card" {
			card, err := readModelCard(root)
			if err != nil {
				return nil, err
			}
			info.Card = card
		}
		if kind == "gguf" {
			gguf, err := InspectGGUF(root)
			if err != nil {
				return nil, err
			}
			info.GGUF = gguf
			applyGGUFModelFields(info, gguf)
		}
		if kind == "safetensors" {
			safetensors, err := InspectSafetensors(root)
			if err != nil {
				return nil, err
			}
			info.Safetensors = safetensors
			info.HFWeights, _ = HFWeightManifestFromInfo(info)
		}
		if kind == "native_manifest" {
			adapter, err := InspectNativeAdapter(root)
			if err != nil {
				return nil, err
			}
			info.NativeAdapter = adapter
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
	info.HFSpec, _ = HFModelSpecFromInfo(info)

	generationConfig, err := readJSON(filepath.Join(root, "generation_config.json"))
	if err != nil {
		return nil, err
	}
	info.GenerationConfig = generationConfig
	info.Generation = ParseGenerationConfig(generationConfig)

	card, err := readModelCard(filepath.Join(root, "README.md"))
	if err != nil {
		return nil, err
	}
	info.Card = card

	files, err := inspectFiles(root)
	if err != nil {
		return nil, err
	}
	info.Files = files
	nativeAdapter, err := InspectNativeAdapter(root)
	if err != nil {
		return nil, err
	}
	info.NativeAdapter = nativeAdapter
	if path := firstFileKind(root, files, "gguf"); path != "" {
		gguf, err := InspectGGUF(path)
		if err != nil {
			return nil, err
		}
		info.GGUF = gguf
		applyGGUFModelFields(info, gguf)
	}
	if safetensors, err := InspectSafetensors(root); err == nil {
		info.Safetensors = safetensors
		info.HFWeights, _ = HFWeightManifestFromInfo(info)
		info.HFShapes, _ = HFShapeReportFromInfo(info)
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

func readModelCard(path string) (*Card, error) {
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxModelCardBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxModelCardBytes {
		return nil, nil
	}
	content := strings.ReplaceAll(string(data), "\r\n", "\n")
	frontMatter, markdown := splitFrontMatter(content)
	card := cardFromFrontMatter(frontMatter)
	card.Title = firstMarkdownHeading(markdown)
	if card.Title == "" && emptyCard(card) {
		return nil, nil
	}
	return card, nil
}

func splitFrontMatter(content string) (string, string) {
	if !strings.HasPrefix(content, "---\n") {
		return "", content
	}
	rest := content[len("---\n"):]
	idx := strings.Index(rest, "\n---")
	if idx < 0 {
		return "", content
	}
	frontMatter := rest[:idx]
	markdown := rest[idx+len("\n---"):]
	markdown = strings.TrimPrefix(markdown, "\n")
	return frontMatter, markdown
}

func cardFromFrontMatter(frontMatter string) *Card {
	values := parseSimpleYAML(frontMatter)
	return &Card{
		License:     firstValue(values["license"]),
		PipelineTag: firstValue(values["pipeline_tag"]),
		LibraryName: firstValue(values["library_name"]),
		Tags:        values["tags"],
		Languages:   values["language"],
		Datasets:    values["datasets"],
		BaseModels:  values["base_model"],
	}
}

func parseSimpleYAML(content string) map[string][]string {
	values := make(map[string][]string)
	var currentKey string
	for _, rawLine := range strings.Split(content, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "- ") && currentKey != "" {
			values[currentKey] = append(values[currentKey], cleanYAMLValue(strings.TrimSpace(strings.TrimPrefix(line, "- "))))
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		currentKey = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if value == "" {
			if _, ok := values[currentKey]; !ok {
				values[currentKey] = nil
			}
			continue
		}
		values[currentKey] = append(values[currentKey], splitYAMLScalar(value)...)
	}
	return values
}

func splitYAMLScalar(value string) []string {
	value = cleanYAMLValue(value)
	if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
		value = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(value, "["), "]"))
		if value == "" {
			return nil
		}
		parts := strings.Split(value, ",")
		out := make([]string, 0, len(parts))
		for _, part := range parts {
			if cleaned := cleanYAMLValue(part); cleaned != "" {
				out = append(out, cleaned)
			}
		}
		return out
	}
	if value == "" {
		return nil
	}
	return []string{value}
}

func cleanYAMLValue(value string) string {
	value = strings.TrimSpace(value)
	value = strings.Trim(value, `"'`)
	return value
}

func firstMarkdownHeading(markdown string) string {
	for _, line := range strings.Split(markdown, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "# "))
		}
	}
	return ""
}

func firstValue(values []string) string {
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func emptyCard(card *Card) bool {
	return card.License == "" &&
		card.PipelineTag == "" &&
		card.LibraryName == "" &&
		len(card.Tags) == 0 &&
		len(card.Languages) == 0 &&
		len(card.Datasets) == 0 &&
		len(card.BaseModels) == 0
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
	base := strings.ToLower(filepath.Base(path))
	lower := strings.ToLower(path)
	switch base {
	case "config.json":
		return "config"
	case "generation_config.json":
		return "generation_config"
	case "manifest.json":
		return "native_manifest"
	case "adapter.json":
		return "native_adapter"
	case "tokenizer.json":
		return "tokenizer"
	case "tokenizer_config.json":
		return "tokenizer_config"
	case "chat_template.jinja":
		return "chat_template"
	case "readme.md":
		return "model_card"
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
