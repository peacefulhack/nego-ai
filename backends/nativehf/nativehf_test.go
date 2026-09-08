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
	core, err := nativeModel.LoadCoreWeights()
	if err != nil {
		t.Fatal(err)
	}
	if core.TokenEmbedding.Tensor.Name != "model.embed_tokens.weight" || len(core.TokenEmbedding.Values) != 1 {
		t.Fatalf("unexpected core weights: %#v", core.TokenEmbedding)
	}
	block, err := nativeModel.LoadBlockWeights(0)
	if err != nil {
		t.Fatal(err)
	}
	if block.Attention.Q.Tensor.Name != "model.layers.0.self_attn.q_proj.weight" || len(block.MLP.Down.Values) != 1 {
		t.Fatalf("unexpected block weights: %#v", block)
	}
	_, err = model.Generate(context.Background(), nego.GenerateRequest{Prompt: "hello"})
	if err == nil || !strings.Contains(err.Error(), "native-hf experimental generation is disabled") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func nativeHFTensorNames() []string {
	return []string{
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
}

func nativeHFSafetensorsFixture(names []string) []byte {
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
	header := b.String()
	data := make([]byte, 8+len(header)+len(names)*4)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	return data
}
