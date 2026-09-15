package modelinfo

import (
	"sort"
	"strings"
)

type CheckReport struct {
	Path            string                  `json:"path"`
	ModelType       string                  `json:"model_type,omitempty"`
	Architecture    string                  `json:"architecture,omitempty"`
	RuntimeFile     *RuntimeFile            `json:"runtime_file,omitempty"`
	ChatTemplate    bool                    `json:"chat_template"`
	ContextLength   uint64                  `json:"context_length,omitempty"`
	Quantization    string                  `json:"quantization,omitempty"`
	Generation      *GenerationConfig       `json:"generation,omitempty"`
	Tokenizer       *TokenizerReport        `json:"tokenizer,omitempty"`
	NativeAdapter   *NativeAdapterInfo      `json:"native_adapter,omitempty"`
	NativeReadiness *NativeRuntimeReadiness `json:"native_readiness,omitempty"`
	Memory          *MemoryEstimate         `json:"memory,omitempty"`
	Artifact        *Artifact               `json:"artifact,omitempty"`
	Backends        []BackendCompatibility  `json:"backends"`
	Warnings        []string                `json:"warnings,omitempty"`
}

type TokenizerReport struct {
	Format                 string  `json:"format,omitempty"`
	Model                  string  `json:"model,omitempty"`
	PreTokenizer           string  `json:"pre_tokenizer,omitempty"`
	VocabSize              uint64  `json:"vocab_size,omitempty"`
	BOSTokenID             *uint32 `json:"bos_token_id,omitempty"`
	EOSTokenID             *uint32 `json:"eos_token_id,omitempty"`
	UNKTokenID             *uint32 `json:"unk_token_id,omitempty"`
	PADTokenID             *uint32 `json:"pad_token_id,omitempty"`
	EOTTokenID             *uint32 `json:"eot_token_id,omitempty"`
	EOMTokenID             *uint32 `json:"eom_token_id,omitempty"`
	AddBOS                 *bool   `json:"add_bos_token,omitempty"`
	AddEOS                 *bool   `json:"add_eos_token,omitempty"`
	AddSpacePrefix         *bool   `json:"add_space_prefix,omitempty"`
	RemoveExtraWhitespaces *bool   `json:"remove_extra_whitespaces,omitempty"`
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
		report.Tokenizer = ggufTokenizerReport(info.GGUF)
	}
	if runtimeFile, err := FindRuntimeFile(info.Path, "gguf", "onnx"); err == nil {
		report.RuntimeFile = runtimeFile
	}
	report.NativeReadiness = nativeRuntimeReadiness(info)
	if memory := estimateMemoryInfo(info, MemoryOptions{}); usefulMemoryEstimate(memory) {
		report.Memory = memory
	}
	report.Backends = backendCompatibility(report, info)
	report.Warnings = checkWarnings(report, info)
	report.Artifact = resolveArtifact(info, report)
	return report, nil
}

func ggufTokenizerReport(info *GGUFInfo) *TokenizerReport {
	if info == nil {
		return nil
	}
	report := &TokenizerReport{
		Format:                 "gguf",
		Model:                  metadataString(info.Metadata, "tokenizer.ggml.model"),
		PreTokenizer:           metadataString(info.Metadata, "tokenizer.ggml.pre"),
		VocabSize:              info.VocabSize,
		BOSTokenID:             metadataOptionalUint32(info.Metadata, "tokenizer.ggml.bos_token_id"),
		EOSTokenID:             metadataOptionalUint32(info.Metadata, "tokenizer.ggml.eos_token_id"),
		UNKTokenID:             metadataOptionalUint32(info.Metadata, "tokenizer.ggml.unknown_token_id"),
		PADTokenID:             metadataOptionalUint32(info.Metadata, "tokenizer.ggml.padding_token_id"),
		EOTTokenID:             metadataOptionalUint32(info.Metadata, "tokenizer.ggml.eot_token_id"),
		EOMTokenID:             metadataOptionalUint32(info.Metadata, "tokenizer.ggml.eom_token_id"),
		AddBOS:                 metadataOptionalBool(info.Metadata, "tokenizer.ggml.add_bos_token"),
		AddEOS:                 metadataOptionalBool(info.Metadata, "tokenizer.ggml.add_eos_token"),
		AddSpacePrefix:         metadataOptionalBool(info.Metadata, "tokenizer.ggml.add_space_prefix"),
		RemoveExtraWhitespaces: metadataOptionalBool(info.Metadata, "tokenizer.ggml.remove_extra_whitespaces"),
	}
	if report.Model == "" && report.PreTokenizer == "" && report.VocabSize == 0 &&
		report.BOSTokenID == nil && report.EOSTokenID == nil && report.UNKTokenID == nil &&
		report.PADTokenID == nil && report.EOTTokenID == nil && report.EOMTokenID == nil &&
		report.AddBOS == nil && report.AddEOS == nil && report.AddSpacePrefix == nil &&
		report.RemoveExtraWhitespaces == nil {
		return nil
	}
	return report
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
	nativeReadiness := report.NativeReadiness
	nativeUnsupported := nativeUnsupportedFromReadiness(nativeReadiness)
	nativeWarnings := nativeCompatibilityWarnings(nativeReadiness)
	nativeCompatible := hasGGUF && nativeReadiness != nil && nativeReadiness.Ready
	out = append(out, BackendCompatibility{
		Name:       "llama.cpp",
		Compatible: hasGGUF,
		Reason:     compatibilityReason(hasGGUF, "GGUF runtime file found", "requires a GGUF file"),
	})
	out = append(out, BackendCompatibility{
		Name:                   "native",
		Compatible:             nativeCompatible,
		Reason:                 nativeCompatibilityReason(hasGGUF, nativeReadiness),
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

func nativeCompatibilityReason(hasGGUF bool, readiness *NativeRuntimeReadiness) string {
	if !hasGGUF {
		return "requires a GGUF file"
	}
	if readiness == nil {
		return "native readiness is unavailable"
	}
	return readiness.Reason
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

func nativeUnsupportedFromReadiness(readiness *NativeRuntimeReadiness) []string {
	if readiness == nil {
		return nil
	}
	return readiness.UnsupportedTensorTypes
}

func nativeCompatibilityWarnings(readiness *NativeRuntimeReadiness) []string {
	if readiness == nil {
		return nil
	}
	return append([]string(nil), readiness.Warnings...)
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
