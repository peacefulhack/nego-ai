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

func TestLoadReadsGGUFWithoutExternalRuntime(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel, ok := model.(*Model)
	if !ok {
		t.Fatalf("unexpected model type %T", model)
	}
	if nativeModel.Info().Architecture != "llama" || len(nativeModel.Info().Tensors) != 1 {
		t.Fatalf("unexpected model info: %#v", nativeModel.Info())
	}
	if nativeModel.Vocab() == nil {
		t.Fatal("expected vocab metadata object")
	}
}

func TestGenerateReportsExperimentalInference(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	_, err = model.Generate(context.Background(), nego.GenerateRequest{Prompt: "hello"})
	if err == nil {
		t.Fatal("expected inference error")
	}
	if !strings.Contains(err.Error(), "native GGUF inference is not implemented yet") || strings.Contains(err.Error(), "llama-cli is not available") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestReadTensorReturnsRawBytes(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	nativeModel := model.(*Model)
	data, tensor, err := nativeModel.ReadTensor("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	if tensor.Type != "f16" {
		t.Fatalf("unexpected tensor type: %s", tensor.Type)
	}
	if len(data) != 256 {
		t.Fatalf("unexpected tensor byte size: %d", len(data))
	}
	if data[0] != 0x7b || data[len(data)-1] != 0x7b {
		t.Fatalf("unexpected tensor data boundaries: %x %x", data[0], data[len(data)-1])
	}
	reader, tensor, err := nativeModel.TensorReader("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	if tensor.Name != "token_embd.weight" || reader.Size() != 256 {
		t.Fatalf("unexpected tensor reader: %s %d", tensor.Name, reader.Size())
	}
	if err := nativeModel.Close(); err != nil {
		t.Fatal(err)
	}
	_, _, err = nativeModel.ReadTensor("token_embd.weight")
	if err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("expected closed tensor store error, got %v", err)
	}
}

func fakeGGUF(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	for _, value := range []any{uint32(3), uint64(1), uint64(3)} {
		if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeStringKV(t, &buf, "general.architecture", "llama")
	writeUint32KV(t, &buf, "general.file_type", 1)
	writeUint32KV(t, &buf, "llama.context_length", 128)
	writeTensor(t, &buf, "token_embd.weight", []uint64{8, 16}, 1, 0)
	padToAlignment(&buf, 32)
	buf.Write(bytes.Repeat([]byte{0x7b}, 256))

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeStringKV(t *testing.T, buf *bytes.Buffer, key, value string) {
	t.Helper()
	writeString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(8)); err != nil {
		t.Fatal(err)
	}
	writeString(t, buf, value)
}

func writeUint32KV(t *testing.T, buf *bytes.Buffer, key string, value uint32) {
	t.Helper()
	writeString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(4)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
		t.Fatal(err)
	}
}

func writeTensor(t *testing.T, buf *bytes.Buffer, name string, shape []uint64, typ uint32, offset uint64) {
	t.Helper()
	writeString(t, buf, name)
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

func padToAlignment(buf *bytes.Buffer, alignment int) {
	for buf.Len()%alignment != 0 {
		buf.WriteByte(0)
	}
}

func writeString(t *testing.T, buf *bytes.Buffer, value string) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	if _, err := buf.WriteString(value); err != nil {
		t.Fatal(err)
	}
}
