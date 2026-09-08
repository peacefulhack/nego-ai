package modelinfo

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEstimateHFMemory(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":4,"max_position_embeddings":8,"hidden_size":4,"num_hidden_layers":1,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2}`)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsShapeFixture(t, hfShapeFixtureTensors(nil)), 0o644); err != nil {
		t.Fatal(err)
	}
	estimate, err := EstimateMemory(dir, MemoryOptions{ContextLength: 4, KVBytes: 4})
	if err != nil {
		t.Fatal(err)
	}
	if estimate.Format != ArtifactFormatHFSafetensors || estimate.WeightBytes == 0 {
		t.Fatalf("unexpected estimate: %#v", estimate)
	}
	wantKV := uint64(4 * 1 * 1 * 2 * 2 * 4)
	if estimate.KVCacheBytes != wantKV || estimate.TotalBytes != estimate.WeightBytes+wantKV {
		t.Fatalf("unexpected KV estimate: %#v want_kv=%d", estimate, wantKV)
	}
}
