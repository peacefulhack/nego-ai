package modelinfo

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSafetensorsStoreLoadsFloat32Tensor(t *testing.T) {
	dir := t.TempDir()
	data := make([]byte, 8)
	binary.LittleEndian.PutUint32(data[0:], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(data[4:], math.Float32bits(-2))
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, safetensorsFixtureWithData(t, `{
		"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
	}`, data), 0o644); err != nil {
		t.Fatal(err)
	}

	store, err := OpenSafetensors(dir)
	if err != nil {
		t.Fatal(err)
	}
	values, tensor, err := store.LoadTensorFloat32("weight")
	if err != nil {
		t.Fatal(err)
	}
	if tensor.Name != "weight" || len(values) != 2 || values[0] != 1.5 || values[1] != -2 {
		t.Fatalf("unexpected values: tensor=%#v values=%#v", tensor, values)
	}
	reader, tensor, err := store.TensorReader("weight")
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if reader.Size() != int64(tensor.ByteSize) {
		t.Fatalf("reader size = %d, want %d", reader.Size(), tensor.ByteSize)
	}
}

func TestSafetensorsStoreLoadsHalfTensors(t *testing.T) {
	dir := t.TempDir()
	f16 := make([]byte, 2)
	binary.LittleEndian.PutUint16(f16, 0x3c00)
	bf16 := make([]byte, 2)
	binary.LittleEndian.PutUint16(bf16, 0x4000)
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, safetensorsFixtureWithData(t, `{
		"half":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},
		"brain":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}
	}`, append(f16, bf16...)), 0o644); err != nil {
		t.Fatal(err)
	}

	store, err := OpenSafetensors(dir)
	if err != nil {
		t.Fatal(err)
	}
	half, _, err := store.LoadTensorFloat32("half")
	if err != nil {
		t.Fatal(err)
	}
	brain, _, err := store.LoadTensorFloat32("brain")
	if err != nil {
		t.Fatal(err)
	}
	if half[0] != 1 || brain[0] != 2 {
		t.Fatalf("unexpected values: half=%#v brain=%#v", half, brain)
	}
}

func safetensorsFixtureWithData(t *testing.T, header string, data []byte) []byte {
	t.Helper()
	out := safetensorsFixture(t, header, len(data))
	copy(out[8+len(headerWithoutOuterSpace(header)):], data)
	return out
}

func headerWithoutOuterSpace(header string) string {
	for len(header) > 0 && (header[0] == ' ' || header[0] == '\n' || header[0] == '\t' || header[0] == '\r') {
		header = header[1:]
	}
	for len(header) > 0 {
		last := header[len(header)-1]
		if last != ' ' && last != '\n' && last != '\t' && last != '\r' {
			break
		}
		header = header[:len(header)-1]
	}
	return header
}
