package training

import (
	"fmt"
	"os"
	"strings"

	"github.com/gakon/nego-ai/modelinfo"
)

type MethodStatus string

const (
	MethodReady       MethodStatus = "ready"
	MethodExternal    MethodStatus = "external"
	MethodPlanned     MethodStatus = "planned"
	MethodUnsupported MethodStatus = "unsupported"
)

type MethodSupport struct {
	Method    string       `json:"method"`
	Status    MethodStatus `json:"status"`
	Available bool         `json:"available"`
	Reason    string       `json:"reason,omitempty"`
}

type Assessment struct {
	BaseModel string              `json:"base_model"`
	Artifact  *modelinfo.Artifact `json:"artifact,omitempty"`
	Methods   []MethodSupport     `json:"methods"`
	Warnings  []string            `json:"warnings,omitempty"`
}

func Assess(baseModel string) (Assessment, error) {
	baseModel = strings.TrimSpace(baseModel)
	report := Assessment{BaseModel: baseModel}
	if baseModel == "" {
		return report, fmt.Errorf("base model is required")
	}
	if _, err := os.Stat(baseModel); err != nil {
		return report, fmt.Errorf("base model %q is not available: %w", baseModel, err)
	}
	artifact, err := modelinfo.Resolve(baseModel)
	if err != nil {
		return report, err
	}
	report.Artifact = artifact
	report.Methods = []MethodSupport{
		tokenBiasSupport(baseModel, artifact),
		nativeLoRASupport(artifact),
		externalProcessSupport(artifact),
		fullFineTuneSupport(artifact),
	}
	if len(artifact.Warnings) > 0 {
		report.Warnings = append(report.Warnings, artifact.Warnings...)
	}
	return report, nil
}

func tokenBiasSupport(baseModel string, artifact *modelinfo.Artifact) MethodSupport {
	support := MethodSupport{
		Method: "token-bias",
		Status: MethodUnsupported,
		Reason: "requires GGUF or Hugging Face safetensors weights plus a tokenizer",
	}
	if artifact == nil || !nativeTrainingFormat(artifact.Format) {
		return support
	}
	if _, err := loadNativeTrainingTokenizer(baseModel, artifact); err != nil {
		support.Reason = err.Error()
		return support
	}
	support.Status = MethodReady
	support.Available = true
	support.Reason = "pure-Go token-bias adapter training is available today"
	return support
}

func nativeLoRASupport(artifact *modelinfo.Artifact) MethodSupport {
	support := MethodSupport{
		Method: "native-lora",
		Status: MethodPlanned,
		Reason: "pure-Go LoRA backprop and optimizer support are not implemented yet",
	}
	if artifact == nil || !nativeTrainingFormat(artifact.Format) {
		support.Status = MethodUnsupported
		support.Reason = "requires trainable local GGUF or Hugging Face safetensors weights"
	}
	return support
}

func externalProcessSupport(artifact *modelinfo.Artifact) MethodSupport {
	support := MethodSupport{
		Method: "external-process",
		Status: MethodUnsupported,
		Reason: "requires a Hugging Face safetensors model directory for common Python training stacks",
	}
	if artifact == nil {
		return support
	}
	switch artifact.Format {
	case modelinfo.ArtifactFormatHFSafetensors, modelinfo.ArtifactFormatMixed:
		support.Status = MethodExternal
		support.Available = true
		support.Reason = "external training jobs can use this Hugging Face-style model directory"
	}
	return support
}

func fullFineTuneSupport(artifact *modelinfo.Artifact) MethodSupport {
	support := MethodSupport{
		Method: "full-finetune",
		Status: MethodPlanned,
		Reason: "full model backprop, optimizer state, and checkpoint writing are planned after LoRA",
	}
	if artifact == nil || !nativeTrainingFormat(artifact.Format) {
		support.Status = MethodUnsupported
		support.Reason = "requires trainable local model weights"
	}
	return support
}
