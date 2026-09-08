package modelinfo

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHFWeightManifestFromSafetensors(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"num_hidden_layers":1}`)
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.q_norm.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.k_norm.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.post_attention_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsNamedFixture(t, names), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := Inspect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.HFWeights == nil || !info.HFWeights.Ready {
		t.Fatalf("unexpected manifest: %#v", info.HFWeights)
	}
	if info.HFWeights.Blocks[0].AttentionQ != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("unexpected block mapping: %#v", info.HFWeights.Blocks[0])
	}
	if info.HFWeights.Blocks[0].AttentionQNorm == "" || info.HFWeights.Blocks[0].AttentionKNorm == "" {
		t.Fatalf("missing optional q/k norm mapping: %#v", info.HFWeights.Blocks[0])
	}
}

func TestHFWeightManifestReportsMissingTensors(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","num_hidden_layers":1}`)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsNamedFixture(t, []string{"model.embed_tokens.weight"}), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, err := BuildHFWeightManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Ready || len(manifest.Missing) == 0 {
		t.Fatalf("expected missing tensors: %#v", manifest)
	}
}

func safetensorsNamedFixture(t *testing.T, names []string) []byte {
	t.Helper()
	var b strings.Builder
	b.WriteString("{")
	for i, name := range names {
		if i > 0 {
			b.WriteString(",")
		}
		begin := i * 4
		end := begin + 4
		fmt.Fprintf(&b, "%q:{%q:%q,%q:[1],%q:[%d,%d]}", name, "dtype", "F32", "shape", "data_offsets", begin, end)
	}
	b.WriteString("}")
	return safetensorsFixture(t, b.String(), len(names)*4)
}
