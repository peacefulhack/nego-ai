package nativehf

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
)

func TestNativeHFBackendLoadsSafetensorsModel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":1,"max_position_embeddings":8,"hidden_size":4,"num_hidden_layers":1,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), nativeHFSafetensorsFixture(nativeHFTensorNames()), 0o644); err != nil {
		t.Fatal(err)
	}

	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{
		Path:    dir,
		Options: map[string]string{"experimental_generation": "false"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	if nativeModel.Info().Safetensors == nil || nativeModel.Spec() == nil || nativeModel.Tokenizer() == nil || nativeModel.Store() == nil {
		t.Fatalf("model did not load native HF components: %#v", nativeModel)
	}
	noCacheModel, err := Backend{}.Load(context.Background(), nego.ModelOptions{
		Path:    dir,
		Options: map[string]string{"experimental_generation": "false", "cache_tensors": "false"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if noCacheModel.(*Model).cache {
		t.Fatal("expected cache_tensors=false to disable native-hf cache")
	}
	_ = noCacheModel.Close()
	core, err := nativeModel.LoadCoreWeights()
	if err != nil {
		t.Fatal(err)
	}
	if core.TokenEmbedding.Tensor.Name != "model.embed_tokens.weight" || len(core.TokenEmbedding.Values) != 4 {
		t.Fatalf("unexpected core weights: %#v", core.TokenEmbedding)
	}
	block, err := nativeModel.LoadBlockWeights(0)
	if err != nil {
		t.Fatal(err)
	}
	if block.Attention.Q.Tensor.Name != "model.layers.0.self_attn.q_proj.weight" || len(block.MLP.Down.Values) != 32 {
		t.Fatalf("unexpected block weights: %#v", block)
	}
	values, _, err := nativeModel.LoadTensorFloat32("model.norm.weight")
	if err != nil {
		t.Fatal(err)
	}
	values[0] = 99
	values, _, err = nativeModel.LoadTensorFloat32("model.norm.weight")
	if err != nil {
		t.Fatal(err)
	}
	if values[0] == 99 {
		t.Fatal("LoadTensorFloat32 should return a copy of cached values")
	}
	_, err = model.Generate(context.Background(), nego.GenerateRequest{Prompt: "hello"})
	if err == nil || !strings.Contains(err.Error(), "native-hf experimental generation is disabled") {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	_, _, err = nativeModel.LoadTensorFloat32("model.norm.weight")
	if err == nil || !strings.Contains(err.Error(), "not loaded") {
		t.Fatalf("expected closed store error, got %v", err)
	}
}

func TestNativeHFBackendRejectsMissingTensors(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":1,"max_position_embeddings":8,"hidden_size":4,"num_hidden_layers":1,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), nativeHFSafetensorsFixture(map[string][]uint64{
		"model.embed_tokens.weight": {1, 4},
	}), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: dir})
	if err == nil || !strings.Contains(err.Error(), "weight manifest is not ready") || !strings.Contains(err.Error(), "model.norm.weight") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func nativeHFTensorNames() map[string][]uint64 {
	return map[string][]uint64{
		"model.embed_tokens.weight":                      {1, 4},
		"model.norm.weight":                              {4},
		"lm_head.weight":                                 {1, 4},
		"model.layers.0.input_layernorm.weight":          {4},
		"model.layers.0.self_attn.q_proj.weight":         {4, 4},
		"model.layers.0.self_attn.q_norm.weight":         {2},
		"model.layers.0.self_attn.k_proj.weight":         {2, 4},
		"model.layers.0.self_attn.k_norm.weight":         {2},
		"model.layers.0.self_attn.v_proj.weight":         {2, 4},
		"model.layers.0.self_attn.o_proj.weight":         {4, 4},
		"model.layers.0.post_attention_layernorm.weight": {4},
		"model.layers.0.mlp.gate_proj.weight":            {8, 4},
		"model.layers.0.mlp.up_proj.weight":              {8, 4},
		"model.layers.0.mlp.down_proj.weight":            {4, 8},
	}
}

func nativeHFSafetensorsFixture(tensors map[string][]uint64) []byte {
	var b strings.Builder
	var payloadBytes int
	b.WriteString("{")
	i := 0
	for name, shape := range tensors {
		if i > 0 {
			b.WriteString(",")
		}
		begin := payloadBytes
		payloadBytes += int(testShapeElementCount(shape) * 4)
		end := payloadBytes
		fmt.Fprintf(&b, "%q:{%q:%q,%q:%s,%q:[%d,%d]}", name, "dtype", "F32", "shape", testShapeJSON(shape), "data_offsets", begin, end)
		i++
	}
	b.WriteString("}")
	header := b.String()
	data := make([]byte, 8+len(header)+payloadBytes)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	return data
}

func testShapeElementCount(shape []uint64) uint64 {
	total := uint64(1)
	for _, dim := range shape {
		total *= dim
	}
	return total
}

func testShapeJSON(shape []uint64) string {
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
