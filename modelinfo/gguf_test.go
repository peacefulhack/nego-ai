package modelinfo

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func TestReadGGUFParsesMetadataSummary(t *testing.T) {
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 2, 7)
	writeGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeGGUFUint32KV(t, &buf, "general.file_type", 15)
	writeGGUFUint32KV(t, &buf, "llama.context_length", 4096)
	writeGGUFUint32KV(t, &buf, "llama.embedding_length", 4096)
	writeGGUFUint32KV(t, &buf, "llama.block_count", 32)
	writeGGUFStringKV(t, &buf, "tokenizer.chat_template", "[INST] {{ message }} [/INST]")
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello", "world"})
	writeGGUFTensor(t, &buf, "token_embd.weight", []uint64{3, 4096}, 1, 0)
	writeGGUFTensor(t, &buf, "blk.0.attn_q.weight", []uint64{4096, 4096}, 12, 8192)

	info, err := ReadGGUF(bytes.NewReader(buf.Bytes()), "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if info.Version != 3 || info.TensorCount != 2 || info.MetadataCount != 7 {
		t.Fatalf("unexpected header: %#v", info)
	}
	if info.Architecture != "llama" || info.Quantization != "mostly_q4_k_m" {
		t.Fatalf("unexpected summary: %#v", info)
	}
	if info.ContextLength != 4096 || info.EmbeddingLength != 4096 || info.BlockCount != 32 || info.VocabSize != 3 {
		t.Fatalf("unexpected dimensions: %#v", info)
	}
	if !strings.Contains(info.ChatTemplate, "[INST]") {
		t.Fatalf("unexpected template: %q", info.ChatTemplate)
	}
	if len(info.Tensors) != 2 {
		t.Fatalf("expected tensor directory, got %#v", info.Tensors)
	}
	if info.Tensors[0].Name != "token_embd.weight" || info.Tensors[0].Type != "f16" || info.Tensors[0].ElementCount != 12288 {
		t.Fatalf("unexpected first tensor: %#v", info.Tensors[0])
	}
	if info.DataOffset == 0 || info.Tensors[1].AbsoluteOffset <= info.Tensors[1].Offset {
		t.Fatalf("unexpected tensor offsets: data=%d tensor=%#v", info.DataOffset, info.Tensors[1])
	}
}

func TestReadGGUFRejectsBadMagic(t *testing.T) {
	_, err := ReadGGUF(strings.NewReader("nope"), "bad.gguf")
	if err == nil {
		t.Fatal("expected invalid magic error")
	}
}

func TestReadGGUFTruncatesStoredArrayValues(t *testing.T) {
	values := make([]string, maxGGUFArrayStored+2)
	for i := range values {
		values[i] = "token"
	}
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 0, 1)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", values)

	info, err := ReadGGUF(bytes.NewReader(buf.Bytes()), "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if info.VocabSize != uint64(len(values)) {
		t.Fatalf("VocabSize = %d, want %d", info.VocabSize, len(values))
	}
	array, ok := info.Metadata["tokenizer.ggml.tokens"].(GGUFArray)
	if !ok {
		t.Fatalf("unexpected array metadata: %#v", info.Metadata["tokenizer.ggml.tokens"])
	}
	if !array.Truncated || len(array.Values) != maxGGUFArrayStored {
		t.Fatalf("array was not truncated correctly: %#v", array)
	}
}

func TestReadGGUFRejectsHugeString(t *testing.T) {
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 0, 1)
	if err := binary.Write(&buf, binary.LittleEndian, uint64(maxGGUFStringLength+1)); err != nil {
		t.Fatal(err)
	}
	_, err := ReadGGUF(bytes.NewReader(buf.Bytes()), "bad.gguf")
	if err == nil {
		t.Fatal("expected huge string error")
	}
}

func writeGGUFHeader(t *testing.T, buf *bytes.Buffer, version uint32, tensors, metadata uint64) {
	t.Helper()
	buf.WriteString("GGUF")
	for _, value := range []any{version, tensors, metadata} {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
}

func writeGGUFStringKV(t *testing.T, buf *bytes.Buffer, key, value string) {
	t.Helper()
	writeGGUFString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(8)); err != nil {
		t.Fatal(err)
	}
	writeGGUFString(t, buf, value)
}

func writeGGUFUint32KV(t *testing.T, buf *bytes.Buffer, key string, value uint32) {
	t.Helper()
	writeGGUFString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(4)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
		t.Fatal(err)
	}
}

func writeGGUFStringArrayKV(t *testing.T, buf *bytes.Buffer, key string, values []string) {
	t.Helper()
	writeGGUFString(t, buf, key)
	for _, value := range []any{uint32(9), uint32(8), uint64(len(values))} {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range values {
		writeGGUFString(t, buf, value)
	}
}

func writeGGUFString(t *testing.T, buf *bytes.Buffer, value string) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	if _, err := buf.WriteString(value); err != nil {
		t.Fatal(err)
	}
}

func writeGGUFTensor(t *testing.T, buf *bytes.Buffer, name string, shape []uint64, typ uint32, offset uint64) {
	t.Helper()
	writeGGUFString(t, buf, name)
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(shape))); err != nil {
		t.Fatal(err)
	}
	for _, dim := range shape {
		if err := binary.Write(buf, binary.LittleEndian, dim); err != nil {
			t.Fatal(err)
		}
	}
	if err := binary.Write(buf, binary.LittleEndian, typ); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(buf, binary.LittleEndian, offset); err != nil {
		t.Fatal(err)
	}
}
