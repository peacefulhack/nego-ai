package nativehf

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
)

func TestForwardTokenTinyHFModel(t *testing.T) {
	dir := writeTinyForwardModel(t)
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	logits, err := nativeModel.ForwardToken(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != 2 {
		t.Fatalf("logits length = %d", len(logits))
	}
	if !(logits[0] > logits[1]) {
		t.Fatalf("expected token 0 to win, logits=%v", logits)
	}
}

func TestForwardTokenWithStateCachesPromptHistory(t *testing.T) {
	dir := writeTinyForwardModel(t)
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	state, err := NewDecodeState(*nativeModel.Spec())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nativeModel.ForwardTokenWithState(0, state); err != nil {
		t.Fatal(err)
	}
	if _, err := nativeModel.ForwardTokenWithState(1, state); err != nil {
		t.Fatal(err)
	}
	if state.Position != 2 {
		t.Fatalf("position = %d, want 2", state.Position)
	}
	if got := state.CachedTokens(0, 0); got != 2 {
		t.Fatalf("cached tokens = %d, want 2", got)
	}
}

func TestGenerateTinyHFModelWhenExplicitlyEnabled(t *testing.T) {
	dir := writeTinyForwardModel(t)
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	out, err := model.Generate(context.Background(), nego.GenerateRequest{Prompt: "a", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if out.Text != "a" {
		t.Fatalf("generated text = %q", out.Text)
	}
}

func TestGenerateTinyHFModelAppliesAdapter(t *testing.T) {
	dir := writeTinyForwardModel(t)
	adapterPath := filepath.Join(t.TempDir(), "adapter.json")
	if err := adapters.Save(adapterPath, &adapters.TokenBiasAdapter{
		Version:   adapters.CurrentVersion,
		Type:      "token_bias",
		VocabSize: 2,
		Bias:      map[int]float32{1: 100},
	}); err != nil {
		t.Fatal(err)
	}
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{
		Path: dir,
		Options: map[string]string{
			"experimental_generation": "true",
			"adapter_path":            adapterPath,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	out, err := model.Generate(context.Background(), nego.GenerateRequest{Prompt: "a", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if out.Text != "b" {
		t.Fatalf("generated text = %q, want adapter-biased token", out.Text)
	}
}

func writeTinyForwardModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":2,"max_position_embeddings":8,"hidden_size":2,"num_hidden_layers":1,"intermediate_size":2,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"a":0,"b":1},"unk_token":"a"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), tinyForwardSafetensorsFixture(t), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

type f32TensorFixture struct {
	shape  []uint64
	values []float32
}

func tinyForwardSafetensorsFixture(t *testing.T) []byte {
	t.Helper()
	identity := []float32{1, 0, 0, 1}
	zeros := []float32{0, 0, 0, 0}
	ones := []float32{1, 1}
	tensors := map[string]f32TensorFixture{
		"model.embed_tokens.weight":                      {shape: []uint64{2, 2}, values: identity},
		"model.norm.weight":                              {shape: []uint64{2}, values: ones},
		"lm_head.weight":                                 {shape: []uint64{2, 2}, values: identity},
		"model.layers.0.input_layernorm.weight":          {shape: []uint64{2}, values: ones},
		"model.layers.0.self_attn.q_proj.weight":         {shape: []uint64{2, 2}, values: identity},
		"model.layers.0.self_attn.q_norm.weight":         {shape: []uint64{2}, values: ones},
		"model.layers.0.self_attn.k_proj.weight":         {shape: []uint64{2, 2}, values: identity},
		"model.layers.0.self_attn.k_norm.weight":         {shape: []uint64{2}, values: ones},
		"model.layers.0.self_attn.v_proj.weight":         {shape: []uint64{2, 2}, values: identity},
		"model.layers.0.self_attn.o_proj.weight":         {shape: []uint64{2, 2}, values: identity},
		"model.layers.0.post_attention_layernorm.weight": {shape: []uint64{2}, values: ones},
		"model.layers.0.mlp.gate_proj.weight":            {shape: []uint64{2, 2}, values: zeros},
		"model.layers.0.mlp.up_proj.weight":              {shape: []uint64{2, 2}, values: zeros},
		"model.layers.0.mlp.down_proj.weight":            {shape: []uint64{2, 2}, values: zeros},
	}
	return f32SafetensorsFixture(t, tensors)
}

func f32SafetensorsFixture(t *testing.T, tensors map[string]f32TensorFixture) []byte {
	t.Helper()
	var header strings.Builder
	var payload []byte
	header.WriteString("{")
	i := 0
	for name, tensor := range tensors {
		if len(tensor.values) != int(shapeElementCount(t, tensor.shape)) {
			t.Fatalf("%s has %d values for shape %v", name, len(tensor.values), tensor.shape)
		}
		if i > 0 {
			header.WriteString(",")
		}
		begin := len(payload)
		for _, value := range tensor.values {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(value))
			payload = append(payload, buf[:]...)
		}
		end := len(payload)
		fmt.Fprintf(&header, "%q:{%q:%q,%q:%s,%q:[%d,%d]}", name, "dtype", "F32", "shape", shapeJSON(tensor.shape), "data_offsets", begin, end)
		i++
	}
	header.WriteString("}")
	out := make([]byte, 8+header.Len()+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(header.Len()))
	copy(out[8:], header.String())
	copy(out[8+header.Len():], payload)
	return out
}

func shapeElementCount(t *testing.T, shape []uint64) uint64 {
	t.Helper()
	total := uint64(1)
	for _, dim := range shape {
		if dim == 0 || total > ^uint64(0)/dim {
			t.Fatalf("invalid shape: %v", shape)
		}
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
