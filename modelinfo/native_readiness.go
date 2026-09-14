package modelinfo

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

type NativeRuntimeReadiness struct {
	Ready                  bool     `json:"ready"`
	Reason                 string   `json:"reason,omitempty"`
	Architecture           string   `json:"architecture,omitempty"`
	RequiredTensorCount    int      `json:"required_tensor_count,omitempty"`
	OptionalTensorCount    int      `json:"optional_tensor_count,omitempty"`
	UnsupportedTensorTypes []string `json:"unsupported_tensor_types,omitempty"`
	UnsupportedRequired    []string `json:"unsupported_required_tensors,omitempty"`
	UnsupportedOptional    []string `json:"unsupported_optional_tensors,omitempty"`
	SpecIssues             []string `json:"spec_issues,omitempty"`
	MissingTensors         []string `json:"missing_tensors,omitempty"`
	ShapeMismatches        []string `json:"shape_mismatches,omitempty"`
	Warnings               []string `json:"warnings,omitempty"`
}

type nativeSpec struct {
	architecture       string
	embeddingLength    uint64
	blockCount         uint64
	feedForwardLength  uint64
	attentionHeadCount uint64
	kvHeadCount        uint64
	ropeTheta          float64
}

type nativeBlockTensorNames struct {
	attentionNorm  string
	attentionQ     string
	attentionQNorm string
	attentionK     string
	attentionKNorm string
	attentionV     string
	attentionOut   string
	ffnNorm        string
	ffnGate        string
	ffnUp          string
	ffnDown        string
}

type nativeTensorNames struct {
	tokenEmbedding string
	outputNorm     string
	output         string
	blocks         []nativeBlockTensorNames
}

func nativeRuntimeReadiness(info *Info) *NativeRuntimeReadiness {
	readiness := &NativeRuntimeReadiness{}
	if info == nil || info.GGUF == nil {
		readiness.Reason = "requires a GGUF file"
		return readiness
	}
	readiness.Architecture = info.GGUF.Architecture
	if len(info.GGUF.Tensors) == 0 {
		readiness.Reason = "requires a GGUF tensor directory"
		return readiness
	}
	readiness.UnsupportedTensorTypes = nativeUnsupportedTensorTypes(info)
	spec, specIssues := buildNativeReadinessSpec(info.GGUF)
	readiness.SpecIssues = specIssues
	if len(specIssues) == 0 {
		names := nativeReadinessTensorNames(spec)
		readiness.RequiredTensorCount = len(nativeRequiredTensorNames(names))
		readiness.OptionalTensorCount = len(nativeOptionalTensorNames(names))
		readiness.UnsupportedRequired = nativeUnsupportedNamedTensors(info.GGUF, nativeRequiredTensorNames(names))
		readiness.UnsupportedOptional = nativeUnsupportedNamedTensors(info.GGUF, nativeOptionalTensorNames(names))
		readiness.MissingTensors = nativeMissingTensors(info.GGUF, names)
		readiness.ShapeMismatches = nativeShapeMismatches(info.GGUF, spec, names)
	}
	readiness.Ready = nativeUnsupportedReady(readiness) &&
		len(readiness.SpecIssues) == 0 &&
		len(readiness.MissingTensors) == 0 &&
		len(readiness.ShapeMismatches) == 0
	readiness.Warnings = nativeReadinessWarnings(info, readiness)
	readiness.Reason = nativeReadinessReason(readiness)
	return readiness
}

func buildNativeReadinessSpec(info *GGUFInfo) (nativeSpec, []string) {
	spec := nativeSpec{
		architecture:       info.Architecture,
		embeddingLength:    info.EmbeddingLength,
		blockCount:         info.BlockCount,
		feedForwardLength:  metadataUint(info.Metadata, nativeMetadataKey(info.Architecture, "feed_forward_length")),
		attentionHeadCount: metadataUint(info.Metadata, nativeMetadataKey(info.Architecture, "attention.head_count")),
		kvHeadCount:        metadataUint(info.Metadata, nativeMetadataKey(info.Architecture, "attention.head_count_kv")),
		ropeTheta:          metadataFloatDefault(info.Metadata, nativeMetadataKey(info.Architecture, "rope.freq_base"), 10000),
	}
	if spec.kvHeadCount == 0 {
		spec.kvHeadCount = spec.attentionHeadCount
	}
	var issues []string
	require := func(ok bool, message string) {
		if !ok {
			issues = append(issues, message)
		}
	}
	require(spec.architecture != "", "missing general.architecture")
	require(spec.embeddingLength > 0, "missing "+nativeMetadataKey(spec.architecture, "embedding_length"))
	require(spec.blockCount > 0, "missing "+nativeMetadataKey(spec.architecture, "block_count"))
	if spec.blockCount > uint64(int(^uint(0)>>1)) {
		issues = append(issues, fmt.Sprintf("block count %d overflows this runtime", spec.blockCount))
	}
	require(spec.feedForwardLength > 0, "missing "+nativeMetadataKey(spec.architecture, "feed_forward_length"))
	require(spec.attentionHeadCount > 0, "missing "+nativeMetadataKey(spec.architecture, "attention.head_count"))
	if spec.embeddingLength > 0 && spec.attentionHeadCount > 0 && spec.embeddingLength%spec.attentionHeadCount != 0 {
		issues = append(issues, fmt.Sprintf("embedding length %d is not divisible by attention head count %d", spec.embeddingLength, spec.attentionHeadCount))
	}
	if spec.attentionHeadCount > 0 && (spec.kvHeadCount == 0 || spec.attentionHeadCount%spec.kvHeadCount != 0) {
		issues = append(issues, fmt.Sprintf("attention head count %d is not divisible by KV head count %d", spec.attentionHeadCount, spec.kvHeadCount))
	}
	if spec.ropeTheta <= 0 || math.IsNaN(spec.ropeTheta) || math.IsInf(spec.ropeTheta, 0) {
		issues = append(issues, "RoPE theta must be finite and positive")
	}
	sort.Strings(issues)
	return spec, issues
}

func nativeMetadataKey(architecture, suffix string) string {
	if architecture == "" {
		return suffix
	}
	return architecture + "." + suffix
}

func nativeReadinessTensorNames(spec nativeSpec) nativeTensorNames {
	out := nativeTensorNames{
		tokenEmbedding: "token_embd.weight",
		outputNorm:     "output_norm.weight",
		output:         "output.weight",
		blocks:         make([]nativeBlockTensorNames, int(spec.blockCount)),
	}
	for i := range out.blocks {
		prefix := fmt.Sprintf("blk.%d.", i)
		out.blocks[i] = nativeBlockTensorNames{
			attentionNorm:  prefix + "attn_norm.weight",
			attentionQ:     prefix + "attn_q.weight",
			attentionQNorm: prefix + "attn_q_norm.weight",
			attentionK:     prefix + "attn_k.weight",
			attentionKNorm: prefix + "attn_k_norm.weight",
			attentionV:     prefix + "attn_v.weight",
			attentionOut:   prefix + "attn_output.weight",
			ffnNorm:        prefix + "ffn_norm.weight",
			ffnGate:        prefix + "ffn_gate.weight",
			ffnUp:          prefix + "ffn_up.weight",
			ffnDown:        prefix + "ffn_down.weight",
		}
	}
	return out
}

func nativeRequiredTensorNames(names nativeTensorNames) []string {
	out := []string{names.tokenEmbedding, names.outputNorm}
	for _, block := range names.blocks {
		out = append(out,
			block.attentionNorm,
			block.attentionQ,
			block.attentionK,
			block.attentionV,
			block.attentionOut,
			block.ffnNorm,
			block.ffnGate,
			block.ffnUp,
			block.ffnDown,
		)
	}
	return out
}

func nativeOptionalTensorNames(names nativeTensorNames) []string {
	out := []string{names.output}
	for _, block := range names.blocks {
		out = append(out, block.attentionQNorm, block.attentionKNorm)
	}
	return out
}

func nativeUnsupportedReady(readiness *NativeRuntimeReadiness) bool {
	if readiness == nil {
		return false
	}
	if len(readiness.SpecIssues) > 0 {
		return len(readiness.UnsupportedTensorTypes) == 0
	}
	return len(readiness.UnsupportedRequired) == 0 && len(readiness.UnsupportedOptional) == 0
}

func nativeUnsupportedNamedTensors(info *GGUFInfo, names []string) []string {
	available := nativeTensorMap(info)
	var out []string
	for _, name := range names {
		tensor, ok := available[name]
		if !ok || nativeSupportsGGMLType(tensor.GGMLType) {
			continue
		}
		out = append(out, tensor.Name+":"+nativeTensorTypeLabel(tensor))
	}
	sort.Strings(out)
	return out
}

func nativeTensorTypeLabel(tensor GGUFTensor) string {
	if tensor.Type != "" {
		return tensor.Type
	}
	return fmt.Sprintf("ggml_type_%d", tensor.GGMLType)
}

func nativeMissingTensors(info *GGUFInfo, names nativeTensorNames) []string {
	available := nativeTensorMap(info)
	var missing []string
	for _, name := range nativeRequiredTensorNames(names) {
		if _, ok := available[name]; !ok {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	return missing
}

func nativeShapeMismatches(info *GGUFInfo, spec nativeSpec, names nativeTensorNames) []string {
	available := nativeTensorMap(info)
	var out []string
	expect := func(name string, shape ...uint64) {
		tensor, ok := available[name]
		if !ok {
			return
		}
		if !nativeSameShape(tensor.Shape, shape) {
			out = append(out, fmt.Sprintf("%s got %v want %v", name, tensor.Shape, shape))
		}
	}
	expect(names.tokenEmbedding, spec.embeddingLength, 0)
	expect(names.outputNorm, spec.embeddingLength)
	if _, ok := available[names.output]; ok {
		expect(names.output, spec.embeddingLength, 0)
	}
	headDim := spec.embeddingLength / spec.attentionHeadCount
	kvProjection := headDim * spec.kvHeadCount
	for _, block := range names.blocks {
		expect(block.attentionNorm, spec.embeddingLength)
		expect(block.attentionQ, spec.embeddingLength, spec.embeddingLength)
		expect(block.attentionQNorm, headDim)
		expect(block.attentionK, spec.embeddingLength, kvProjection)
		expect(block.attentionKNorm, headDim)
		expect(block.attentionV, spec.embeddingLength, kvProjection)
		expect(block.attentionOut, spec.embeddingLength, spec.embeddingLength)
		expect(block.ffnNorm, spec.embeddingLength)
		expect(block.ffnGate, spec.embeddingLength, spec.feedForwardLength)
		expect(block.ffnUp, spec.embeddingLength, spec.feedForwardLength)
		expect(block.ffnDown, spec.feedForwardLength, spec.embeddingLength)
	}
	sort.Strings(out)
	return out
}

func nativeTensorMap(info *GGUFInfo) map[string]GGUFTensor {
	available := make(map[string]GGUFTensor, len(info.Tensors))
	for _, tensor := range info.Tensors {
		available[tensor.Name] = tensor
	}
	return available
}

func nativeSameShape(got []uint64, want []uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if want[i] != 0 && got[i] != want[i] {
			return false
		}
	}
	return true
}

func nativeReadinessWarnings(info *Info, readiness *NativeRuntimeReadiness) []string {
	var warnings []string
	if readiness == nil {
		return warnings
	}
	if len(readiness.UnsupportedRequired) > 0 || len(readiness.UnsupportedOptional) > 0 ||
		(len(readiness.SpecIssues) > 0 && len(readiness.UnsupportedTensorTypes) > 0) {
		warnings = append(warnings, "native pure-Go runtime cannot load all tensor types required by this model yet")
	}
	if info != nil && info.GGUF != nil && strings.Contains(strings.ToLower(info.GGUF.Quantization), "_k") {
		warnings = append(warnings, "K-quant GGUF support is experimental; validate outputs before relying on them")
	}
	if readiness.Ready {
		warnings = append(warnings, "native pure-Go generation is experimental and not production-quality yet")
	}
	return warnings
}

func nativeReadinessReason(readiness *NativeRuntimeReadiness) string {
	switch {
	case readiness == nil:
		return "native readiness is unavailable"
	case readiness.Ready:
		return "GGUF metadata, tensor manifest, shapes, and tensor types are ready for the experimental pure-Go backend"
	case len(readiness.SpecIssues) > 0:
		return "native GGUF spec is incomplete: " + readiness.SpecIssues[0]
	case len(readiness.MissingTensors) > 0:
		return fmt.Sprintf("native GGUF tensor manifest is incomplete: %d required tensors missing", len(readiness.MissingTensors))
	case len(readiness.ShapeMismatches) > 0:
		return fmt.Sprintf("native GGUF tensor shapes do not match expected Llama/Qwen layout: %d mismatches", len(readiness.ShapeMismatches))
	case len(readiness.UnsupportedRequired) > 0:
		return fmt.Sprintf("unsupported required tensor types: %d tensors", len(readiness.UnsupportedRequired))
	case len(readiness.UnsupportedOptional) > 0:
		return fmt.Sprintf("unsupported optional tensor types: %d tensors", len(readiness.UnsupportedOptional))
	case len(readiness.UnsupportedTensorTypes) > 0:
		return "unsupported tensor types: " + strings.Join(readiness.UnsupportedTensorTypes, ", ")
	default:
		return "native GGUF artifact is not ready"
	}
}

func metadataFloatDefault(metadata map[string]any, key string, fallback float64) float64 {
	switch value := metadata[key].(type) {
	case uint8:
		return float64(value)
	case uint16:
		return float64(value)
	case uint32:
		return float64(value)
	case uint64:
		return float64(value)
	case int8:
		return float64(value)
	case int16:
		return float64(value)
	case int32:
		return float64(value)
	case int64:
		return float64(value)
	case float32:
		return float64(value)
	case float64:
		return value
	default:
		return fallback
	}
}
