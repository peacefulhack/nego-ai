package modelinfo

import (
	"sort"
	"strings"
)

type CheckReport struct {
	Path          string                 `json:"path"`
	ModelType     string                 `json:"model_type,omitempty"`
	Architecture  string                 `json:"architecture,omitempty"`
	RuntimeFile   *RuntimeFile           `json:"runtime_file,omitempty"`
	ChatTemplate  bool                   `json:"chat_template"`
	ContextLength uint64                 `json:"context_length,omitempty"`
	Quantization  string                 `json:"quantization,omitempty"`
	Generation    *GenerationConfig      `json:"generation,omitempty"`
	NativeAdapter *NativeAdapterInfo     `json:"native_adapter,omitempty"`
	Artifact      *Artifact              `json:"artifact,omitempty"`
	Backends      []BackendCompatibility `json:"backends"`
	Warnings      []string               `json:"warnings,omitempty"`
}

type BackendCompatibility struct {
	Name                   string   `json:"name"`
	Compatible             bool     `json:"compatible"`
	Reason                 string   `json:"reason,omitempty"`
	UnsupportedTensorTypes []string `json:"unsupported_tensor_types,omitempty"`
	Warnings               []string `json:"warnings,omitempty"`
}

func Check(path string) (*CheckReport, error) {
	info, err := Inspect(path)
	if err != nil {
		return nil, err
	}
	return checkInfo(info)
}

func checkInfo(info *Info) (*CheckReport, error) {
	report := &CheckReport{
		Path:          info.Path,
		ModelType:     info.ModelType,
		ChatTemplate:  hasChatTemplate(info),
		Generation:    info.Generation,
		NativeAdapter: info.NativeAdapter,
	}
	if len(info.Architectures) > 0 {
		report.Architecture = info.Architectures[0]
	}
	if info.GGUF != nil {
		report.ContextLength = info.GGUF.ContextLength
		report.Quantization = info.GGUF.Quantization
	}
	if runtimeFile, err := FindRuntimeFile(info.Path, "gguf", "onnx"); err == nil {
		report.RuntimeFile = runtimeFile
	}
	report.Backends = backendCompatibility(report, info)
	report.Warnings = checkWarnings(report, info)
	report.Artifact = resolveArtifact(info, report)
	return report, nil
}

func hasChatTemplate(info *Info) bool {
	if info.GGUF != nil && info.GGUF.ChatTemplate != "" {
		return true
	}
	for _, file := range info.Files {
		if file.Kind == "chat_template" || file.Kind == "tokenizer_config" {
			return true
		}
	}
	return false
}

func backendCompatibility(report *CheckReport, info *Info) []BackendCompatibility {
	if info.NativeAdapter != nil {
		name := firstNonEmpty(info.NativeAdapter.RecommendedBackend, "native")
		return []BackendCompatibility{{
			Name:       name,
			Compatible: true,
			Reason:     "native training output manifest points to a base model and adapter",
		}}
	}
	var out []BackendCompatibility
	hasGGUF := report.RuntimeFile != nil && report.RuntimeFile.Kind == "gguf"
	hasONNX := report.RuntimeFile != nil && report.RuntimeFile.Kind == "onnx"
	hasSafetensors := false
	for _, file := range info.Files {
		if file.Kind == "safetensors" {
			hasSafetensors = true
			break
		}
	}
	nativeHFCompatible, nativeHFReason := nativeHFCompatibility(info, hasSafetensors)
	nativeUnsupported := nativeUnsupportedTensorTypes(info)
	nativeWarnings := nativeCompatibilityWarnings(report, nativeUnsupported)
	nativeHasTensors := info.GGUF != nil && len(info.GGUF.Tensors) > 0
	nativeCompatible := hasGGUF && nativeHasTensors && len(nativeUnsupported) == 0
	out = append(out, BackendCompatibility{
		Name:       "llama.cpp",
		Compatible: hasGGUF,
		Reason:     compatibilityReason(hasGGUF, "GGUF runtime file found", "requires a GGUF file"),
	})
	out = append(out, BackendCompatibility{
		Name:                   "native",
		Compatible:             nativeCompatible,
		Reason:                 nativeCompatibilityReason(hasGGUF, nativeHasTensors, nativeUnsupported),
		UnsupportedTensorTypes: nativeUnsupported,
		Warnings:               nativeWarnings,
	})
	out = append(out, BackendCompatibility{
		Name:       "onnx",
		Compatible: hasONNX,
		Reason:     compatibilityReason(hasONNX, "ONNX runtime file found", "requires an ONNX file"),
	})
	out = append(out, BackendCompatibility{
		Name:       "native-hf",
		Compatible: nativeHFCompatible,
		Reason:     nativeHFReason,
	})
	return out
}

func compatibilityReason(ok bool, good, bad string) string {
	if ok {
		return good
	}
	return bad
}

func nativeCompatibilityReason(hasGGUF, hasTensors bool, unsupported []string) string {
	if !hasGGUF {
		return "requires a GGUF file"
	}
	if !hasTensors {
		return "requires a GGUF tensor directory"
	}
	if len(unsupported) > 0 {
		return "unsupported tensor types: " + strings.Join(unsupported, ", ")
	}
	return "GGUF tensor types can be loaded by the experimental pure-Go backend"
}

func nativeHFCompatibility(info *Info, hasSafetensors bool) (bool, string) {
	if !hasSafetensors {
		return false, "requires safetensors weights"
	}
	if !hasTokenizerFiles(info.Files) {
		return false, "requires tokenizer.json"
	}
	if info.HFSpec == nil {
		return false, "requires Hugging Face config.json model spec"
	}
	if !info.HFSpec.Ready {
		return false, firstNonEmpty(info.HFSpec.ValidationError, "Hugging Face model spec is incomplete")
	}
	if info.HFWeights == nil {
		return false, "requires a Hugging Face weight manifest"
	}
	if !info.HFWeights.Ready {
		return false, "Hugging Face weight manifest is missing tensors"
	}
	if info.HFShapes != nil && !info.HFShapes.Ready {
		return false, "Hugging Face tensor shapes do not match config"
	}
	return true, "native-hf can run this safetensors model through the experimental pure-Go path"
}

func nativeUnsupportedTensorTypes(info *Info) []string {
	if info.GGUF == nil {
		return nil
	}
	seen := make(map[string]bool)
	for _, tensor := range info.GGUF.Tensors {
		if !nativeSupportsGGMLType(tensor.GGMLType) {
			typ := tensor.Type
			if typ == "" {
				typ = "ggml_type"
			}
			seen[typ] = true
		}
	}
	out := make([]string, 0, len(seen))
	for typ := range seen {
		out = append(out, typ)
	}
	sort.Strings(out)
	return out
}

func nativeSupportsGGMLType(typ uint32) bool {
	switch typ {
	case 0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30:
		return true
	default:
		return false
	}
}

func nativeCompatibilityWarnings(report *CheckReport, unsupported []string) []string {
	if len(unsupported) == 0 {
		return nil
	}
	warnings := []string{"native pure-Go runtime cannot load all tensor types yet"}
	if strings.Contains(strings.ToLower(report.Quantization), "_k") {
		warnings = append(warnings, "K-quant GGUF models still need broader native kernels")
	}
	return warnings
}

func checkWarnings(report *CheckReport, info *Info) []string {
	var warnings []string
	if info.NativeAdapter == nil && report.RuntimeFile == nil {
		warnings = append(warnings, "no local runtime file found")
	}
	if info.NativeAdapter == nil && !report.ChatTemplate {
		warnings = append(warnings, "no chat template detected")
	}
	if info.NativeAdapter == nil && info.ModelType == "" && report.Architecture == "" {
		warnings = append(warnings, "model architecture metadata is missing")
	}
	for _, backend := range report.Backends {
		if backend.Name == "native" && !backend.Compatible && len(backend.UnsupportedTensorTypes) > 0 {
			warnings = append(warnings, "native backend unsupported tensor types: "+strings.Join(backend.UnsupportedTensorTypes, ", "))
		}
	}
	return warnings
}
