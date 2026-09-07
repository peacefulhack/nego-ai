package nativehf

import (
	"context"
	"encoding/binary"
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
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), nativeHFSafetensorsFixture(), 0o644); err != nil {
		t.Fatal(err)
	}

	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	if nativeModel.Info().Safetensors == nil || nativeModel.Spec() == nil || nativeModel.Tokenizer() == nil || nativeModel.Store() == nil {
		t.Fatalf("model did not load native HF components: %#v", nativeModel)
	}
	_, err = model.Generate(context.Background(), nego.GenerateRequest{Prompt: "hello"})
	if err == nil || !strings.Contains(err.Error(), "native-hf inference is not implemented yet") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func nativeHFSafetensorsFixture() []byte {
	header := `{"model.embed_tokens.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	data := make([]byte, 8+len(header)+4)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	return data
}
