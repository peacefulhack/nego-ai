package modelinfo

import (
	"fmt"
	"strings"
)

type ArtifactFormat string

const (
	ArtifactFormatUnknown       ArtifactFormat = "unknown"
	ArtifactFormatGGUF          ArtifactFormat = "gguf"
	ArtifactFormatHFSafetensors ArtifactFormat = "hf-safetensors"
	ArtifactFormatONNX          ArtifactFormat = "onnx"
	ArtifactFormatMixed         ArtifactFormat = "mixed"
	ArtifactFormatMetadataOnly  ArtifactFormat = "metadata-only"
)

type ArtifactCapabilityStatus string

const (
	CapabilityReady        ArtifactCapabilityStatus = "ready"
	CapabilityExperimental ArtifactCapabilityStatus = "experimental"
	CapabilityExternal     ArtifactCapabilityStatus = "external"
	CapabilityPlanned      ArtifactCapabilityStatus = "planned"
	CapabilityUnsupported  ArtifactCapabilityStatus = "unsupported"
)

type Artifact struct {
	Path                    string               `json:"path"`
	Format                  ArtifactFormat       `json:"format"`
	ModelType               string               `json:"model_type,omitempty"`
	Architectures           []string             `json:"architectures,omitempty"`
	RuntimeFile             *RuntimeFile         `json:"runtime_file,omitempty"`
	ChatTemplate            bool                 `json:"chat_template"`
	ContextLength           uint64               `json:"context_length,omitempty"`
	Quantization            string               `json:"quantization,omitempty"`
	ParameterCount          uint64               `json:"parameter_count,omitempty"`
	TensorBytes             uint64               `json:"tensor_bytes,omitempty"`
	RecommendedRunBackend   string               `json:"recommended_run_backend,omitempty"`
	RecommendedTrainBackend string               `json:"recommended_train_backend,omitempty"`
	RunBackends             []ArtifactCapability `json:"run_backends,omitempty"`
	TrainBackends           []ArtifactCapability `json:"train_backends,omitempty"`
	Warnings                []string             `json:"warnings,omitempty"`
}

type ArtifactCapability struct {
	Name   string                   `json:"name"`
	Status ArtifactCapabilityStatus `json:"status"`
	Reason string                   `json:"reason,omitempty"`
}

func Resolve(path string) (*Artifact, error) {
	info, err := Inspect(path)
	if err != nil {
		return nil, err
	}
	report, err := checkInfo(info)
	if err != nil {
		return nil, err
	}
	return resolveArtifact(info, report), nil
}

func resolveArtifact(info *Info, report *CheckReport) *Artifact {
	artifact := &Artifact{
		Path:          info.Path,
		Format:        artifactFormat(info),
		ModelType:     info.ModelType,
		Architectures: append([]string(nil), info.Architectures...),
		ChatTemplate:  hasChatTemplate(info),
	}
	if report != nil {
		artifact.RuntimeFile = report.RuntimeFile
		artifact.ContextLength = report.ContextLength
		artifact.Quantization = report.Quantization
	}
	if info.GGUF != nil {
		artifact.ParameterCount = ggufParameterCount(info.GGUF)
	}
	if info.Safetensors != nil {
		artifact.ParameterCount = info.Safetensors.ParamCount
		artifact.TensorBytes = info.Safetensors.TotalSize
	}
	artifact.RunBackends = runCapabilities(artifact, report)
	artifact.TrainBackends = trainCapabilities(artifact)
	artifact.RecommendedRunBackend = recommendedCapability(artifact.RunBackends)
	artifact.RecommendedTrainBackend = recommendedCapability(artifact.TrainBackends)
	artifact.Warnings = artifactWarnings(artifact, report)
	return artifact
}

func artifactFormat(info *Info) ArtifactFormat {
	hasGGUF := info.GGUF != nil || hasFileKindValue(info.Files, "gguf")
	hasSafetensors := info.Safetensors != nil || hasFileKindValue(info.Files, "safetensors")
	hasONNX := hasFileKindValue(info.Files, "onnx")
	count := boolCount(hasGGUF, hasSafetensors, hasONNX)
	if count > 1 {
		return ArtifactFormatMixed
	}
	switch {
	case hasGGUF:
		return ArtifactFormatGGUF
	case hasSafetensors:
		return ArtifactFormatHFSafetensors
	case hasONNX:
		return ArtifactFormatONNX
	case info.Config != nil || info.GenerationConfig != nil || info.Card != nil || hasTokenizerFiles(info.Files):
		return ArtifactFormatMetadataOnly
	default:
		return ArtifactFormatUnknown
	}
}

func runCapabilities(artifact *Artifact, report *CheckReport) []ArtifactCapability {
	var out []ArtifactCapability
	switch artifact.Format {
	case ArtifactFormatGGUF, ArtifactFormatMixed:
		native := backendCompatibilityByName(report, "native")
		if native.Compatible {
			out = append(out, ArtifactCapability{
				Name:   "native",
				Status: CapabilityExperimental,
				Reason: "pure-Go GGUF runtime can load this artifact; production generation is still experimental",
			})
		} else {
			out = append(out, ArtifactCapability{
				Name:   "native",
				Status: CapabilityUnsupported,
				Reason: firstNonEmpty(native.Reason, "pure-Go GGUF runtime cannot load this artifact yet"),
			})
		}
		out = append(out, ArtifactCapability{
			Name:   "llama.cpp",
			Status: CapabilityExternal,
			Reason: "GGUF runtime file can be run through the optional llama.cpp backend",
		})
	case ArtifactFormatHFSafetensors:
		out = append(out, ArtifactCapability{
			Name:   "native-hf",
			Status: CapabilityPlanned,
			Reason: "native-hf can inspect and load safetensors models, but production autoregressive generation is still guarded",
		})
	case ArtifactFormatONNX:
		out = append(out, ArtifactCapability{
			Name:   "onnx",
			Status: CapabilityPlanned,
			Reason: "ONNX runtime backend is not implemented yet",
		})
	default:
		out = append(out, ArtifactCapability{
			Name:   "runtime",
			Status: CapabilityUnsupported,
			Reason: "no runnable local model weights detected",
		})
	}
	return out
}

func trainCapabilities(artifact *Artifact) []ArtifactCapability {
	switch artifact.Format {
	case ArtifactFormatHFSafetensors, ArtifactFormatMixed:
		return []ArtifactCapability{
			{
				Name:   "native-token-bias",
				Status: CapabilityExperimental,
				Reason: "pure-Go token-bias adapter training can use this model vocabulary",
			},
			{
				Name:   "process",
				Status: CapabilityExternal,
				Reason: "Hugging Face-style weights can be used as a base model for external training jobs",
			},
			{
				Name:   "native-lora",
				Status: CapabilityPlanned,
				Reason: "pure-Go LoRA fine-tuning over safetensors is not implemented yet",
			},
		}
	case ArtifactFormatGGUF:
		return []ArtifactCapability{
			{
				Name:   "native-token-bias",
				Status: CapabilityExperimental,
				Reason: "pure-Go token-bias adapter training can use this GGUF or sidecar vocabulary",
			},
			{
				Name:   "native-lora",
				Status: CapabilityPlanned,
				Reason: "pure-Go LoRA fine-tuning for GGUF runtime artifacts is not implemented yet",
			},
			{
				Name:   "hf-trainer",
				Status: CapabilityUnsupported,
				Reason: "most training stacks expect Hugging Face safetensors, not GGUF runtime files",
			},
		}
	default:
		return []ArtifactCapability{
			{
				Name:   "trainer",
				Status: CapabilityUnsupported,
				Reason: "no trainable local model weights detected",
			},
		}
	}
}

func recommendedCapability(capabilities []ArtifactCapability) string {
	for _, status := range []ArtifactCapabilityStatus{CapabilityReady, CapabilityExperimental, CapabilityExternal} {
		for _, capability := range capabilities {
			if capability.Status == status {
				return capability.Name
			}
		}
	}
	return ""
}

func artifactWarnings(artifact *Artifact, report *CheckReport) []string {
	seen := make(map[string]bool)
	var warnings []string
	add := func(warning string) {
		warning = strings.TrimSpace(warning)
		if warning == "" || seen[warning] {
			return
		}
		seen[warning] = true
		warnings = append(warnings, warning)
	}
	if report != nil {
		for _, warning := range report.Warnings {
			add(warning)
		}
	}
	if artifact.RecommendedRunBackend == "" {
		add("no runnable backend is ready for this artifact yet")
	}
	if artifact.RecommendedTrainBackend == "" {
		add("no training backend is ready for this artifact yet")
	}
	if artifact.Format == ArtifactFormatHFSafetensors {
		add("downloaded Hugging Face safetensors can be inspected, loaded, and used for native token-bias training, but production native chat is still planned")
	}
	return warnings
}

func hasFileKindValue(files []File, kind string) bool {
	for _, file := range files {
		if file.Kind == kind {
			return true
		}
	}
	return false
}

func hasTokenizerFiles(files []File) bool {
	return hasFileKindValue(files, "tokenizer") || hasFileKindValue(files, "tokenizer_config")
}

func boolCount(values ...bool) int {
	count := 0
	for _, value := range values {
		if value {
			count++
		}
	}
	return count
}

func backendCompatibilityByName(report *CheckReport, name string) BackendCompatibility {
	if report == nil {
		return BackendCompatibility{}
	}
	for _, backend := range report.Backends {
		if backend.Name == name {
			return backend
		}
	}
	return BackendCompatibility{}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func ggufParameterCount(info *GGUFInfo) uint64 {
	var count uint64
	for _, tensor := range info.Tensors {
		n := tensor.ElementCount
		if n == 0 {
			computed, err := shapeElementCount(tensor.Shape)
			if err != nil {
				return ^uint64(0)
			}
			n = computed
		}
		if count > ^uint64(0)-n {
			return ^uint64(0)
		}
		count += n
	}
	return count
}

func FormatResolveError(path string, artifact *Artifact) error {
	if artifact == nil {
		return fmt.Errorf("no runnable model backend found for %q", path)
	}
	reason := "no runnable backend is ready"
	for _, capability := range artifact.RunBackends {
		if capability.Reason != "" {
			reason = capability.Reason
			break
		}
	}
	return fmt.Errorf("no runnable model backend found for %q (%s): %s", path, artifact.Format, reason)
}
