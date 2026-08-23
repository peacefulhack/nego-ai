package native

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
)

func TestLoadBlockWeights(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeBlockGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	weights, err := nativeModel.LoadBlockWeights(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.AttentionNorm) != 2 || len(weights.Attention.QValues) != 4 || weights.Attention.QTensor.Name != "blk.0.attn_q.weight" {
		t.Fatalf("unexpected weights: %#v", weights)
	}
}

func TestLoadBlockWeightsRejectsMissingTensor(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	_, err = nativeModel.LoadBlockWeights(0)
	if err == nil || !strings.Contains(err.Error(), "attn_norm") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestForwardToken(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeBlockGGUF(t)})
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
		t.Fatalf("unexpected logits: %#v", logits)
	}
}

func TestForwardTokenRejectsUnreadyManifest(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	_, err = nativeModel.ForwardToken(0, 0)
	if err == nil || !strings.Contains(err.Error(), "manifest") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func fakeBlockGGUF(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	tensorCount := uint64(12)
	metadataCount := uint64(8)
	for _, value := range []any{uint32(3), tensorCount, metadataCount} {
		if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeStringKV(t, &buf, "general.architecture", "llama")
	writeUint32KV(t, &buf, "general.file_type", 0)
	writeUint32KV(t, &buf, "llama.context_length", 8)
	writeUint32KV(t, &buf, "llama.embedding_length", 2)
	writeUint32KV(t, &buf, "llama.block_count", 1)
	writeUint32KV(t, &buf, "llama.feed_forward_length", 2)
	writeUint32KV(t, &buf, "llama.attention.head_count", 1)
	writeUint32KV(t, &buf, "llama.attention.head_count_kv", 1)

	offset := uint64(0)
	writeF32TensorDir(t, &buf, "token_embd.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "output_norm.weight", []uint64{2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.attn_norm.weight", []uint64{2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.attn_q.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.attn_k.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.attn_v.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.attn_output.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.ffn_norm.weight", []uint64{2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.ffn_gate.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.ffn_up.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "blk.0.ffn_down.weight", []uint64{2, 2}, &offset)
	writeF32TensorDir(t, &buf, "output.weight", []uint64{2, 2}, &offset)

	padToAlignment(&buf, 32)
	for i := uint64(0); i < offset/4; i++ {
		if err := binary.Write(&buf, binary.LittleEndian, float32(1)); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeF32TensorDir(t *testing.T, buf *bytes.Buffer, name string, shape []uint64, offset *uint64) {
	t.Helper()
	writeTensor(t, buf, name, shape, 0, *offset)
	elements := uint64(1)
	for _, dim := range shape {
		elements *= dim
	}
	*offset += elements * 4
}
