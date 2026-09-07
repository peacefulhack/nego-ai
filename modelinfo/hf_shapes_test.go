package modelinfo

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHFShapeReportReady(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":4,"max_position_embeddings":8,"hidden_size":4,"num_hidden_layers":1,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2}`)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsShapeFixture(t, hfShapeFixtureTensors(nil)), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := Inspect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.HFShapes == nil || !info.HFShapes.Ready {
		t.Fatalf("unexpected HF shapes: %#v", info.HFShapes)
	}
	if len(info.HFShapes.Checked) != len(hfShapeFixtureTensors(nil)) {
		t.Fatalf("checked tensors = %d", len(info.HFShapes.Checked))
	}
}

func TestHFShapeReportMismatch(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":4,"max_position_embeddings":8,"hidden_size":4,"num_hidden_layers":1,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2}`)
	override := map[string][]uint64{"model.layers.0.self_attn.k_proj.weight": {4, 4}}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsShapeFixture(t, hfShapeFixtureTensors(override)), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := Inspect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.HFShapes == nil || info.HFShapes.Ready || len(info.HFShapes.MissingShape) != 1 {
		t.Fatalf("unexpected HF shapes: %#v", info.HFShapes)
	}
	if !strings.Contains(info.HFShapes.MissingShape[0], "k_proj") {
		t.Fatalf("unexpected mismatch: %#v", info.HFShapes.MissingShape)
	}
}

func hfShapeFixtureTensors(overrides map[string][]uint64) map[string][]uint64 {
	tensors := map[string][]uint64{
		"model.embed_tokens.weight":                      {4, 4},
		"model.norm.weight":                              {4},
		"lm_head.weight":                                 {4, 4},
		"model.layers.0.input_layernorm.weight":          {4},
		"model.layers.0.self_attn.q_proj.weight":         {4, 4},
		"model.layers.0.self_attn.k_proj.weight":         {2, 4},
		"model.layers.0.self_attn.v_proj.weight":         {2, 4},
		"model.layers.0.self_attn.o_proj.weight":         {4, 4},
		"model.layers.0.post_attention_layernorm.weight": {4},
		"model.layers.0.mlp.gate_proj.weight":            {8, 4},
		"model.layers.0.mlp.up_proj.weight":              {8, 4},
		"model.layers.0.mlp.down_proj.weight":            {4, 8},
	}
	for name, shape := range overrides {
		tensors[name] = shape
	}
	return tensors
}

func safetensorsShapeFixture(t *testing.T, tensors map[string][]uint64) []byte {
	t.Helper()
	var b strings.Builder
	b.WriteString("{")
	offset := uint64(0)
	i := 0
	for name, shape := range tensors {
		if i > 0 {
			b.WriteString(",")
		}
		size := shapeParams(shape) * 4
		fmt.Fprintf(&b, "%q:{%q:%q,%q:%s,%q:[%d,%d]}", name, "dtype", "F32", "shape", shapeJSON(shape), "data_offsets", offset, offset+size)
		offset += size
		i++
	}
	b.WriteString("}")
	return safetensorsFixture(t, b.String(), int(offset))
}

func shapeParams(shape []uint64) uint64 {
	total := uint64(1)
	for _, dim := range shape {
		total *= dim
	}
	return total
}

func shapeJSON(shape []uint64) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, dim := range shape {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%d", dim)
	}
	b.WriteByte(']')
	return b.String()
}
