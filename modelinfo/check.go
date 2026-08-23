package modelinfo

type CheckReport struct {
	Path          string                 `json:"path"`
	ModelType     string                 `json:"model_type,omitempty"`
	Architecture  string                 `json:"architecture,omitempty"`
	RuntimeFile   *RuntimeFile           `json:"runtime_file,omitempty"`
	ChatTemplate  bool                   `json:"chat_template"`
	ContextLength uint64                 `json:"context_length,omitempty"`
	Quantization  string                 `json:"quantization,omitempty"`
	Backends      []BackendCompatibility `json:"backends"`
	Warnings      []string               `json:"warnings,omitempty"`
}

type BackendCompatibility struct {
	Name       string `json:"name"`
	Compatible bool   `json:"compatible"`
	Reason     string `json:"reason,omitempty"`
}

func Check(path string) (*CheckReport, error) {
	info, err := Inspect(path)
	if err != nil {
		return nil, err
	}
	report := &CheckReport{
		Path:         info.Path,
		ModelType:    info.ModelType,
		ChatTemplate: hasChatTemplate(info),
	}
	if len(info.Architectures) > 0 {
		report.Architecture = info.Architectures[0]
	}
	if info.GGUF != nil {
		report.ContextLength = info.GGUF.ContextLength
		report.Quantization = info.GGUF.Quantization
	}
	if runtimeFile, err := FindRuntimeFile(path, "gguf", "onnx"); err == nil {
		report.RuntimeFile = runtimeFile
	}
	report.Backends = backendCompatibility(report, info)
	report.Warnings = checkWarnings(report, info)
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
	out = append(out, BackendCompatibility{
		Name:       "llama.cpp",
		Compatible: hasGGUF,
		Reason:     compatibilityReason(hasGGUF, "GGUF runtime file found", "requires a GGUF file"),
	})
	out = append(out, BackendCompatibility{
		Name:       "onnx",
		Compatible: hasONNX,
		Reason:     compatibilityReason(hasONNX, "ONNX runtime file found", "requires an ONNX file"),
	})
	out = append(out, BackendCompatibility{
		Name:       "hf-safetensors",
		Compatible: hasSafetensors,
		Reason:     compatibilityReason(hasSafetensors, "safetensors weights found", "requires safetensors weights"),
	})
	return out
}

func compatibilityReason(ok bool, good, bad string) string {
	if ok {
		return good
	}
	return bad
}

func checkWarnings(report *CheckReport, info *Info) []string {
	var warnings []string
	if report.RuntimeFile == nil {
		warnings = append(warnings, "no local runtime file found")
	}
	if !report.ChatTemplate {
		warnings = append(warnings, "no chat template detected")
	}
	if info.ModelType == "" && report.Architecture == "" {
		warnings = append(warnings, "model architecture metadata is missing")
	}
	return warnings
}
