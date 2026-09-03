package modelinfo

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadSafetensorsHeader(t *testing.T) {
	data := safetensorsFixture(t, `{
		"__metadata__":{"format":"pt"},
		"model.embed_tokens.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]},
		"lm_head.weight":{"dtype":"F32","shape":[3],"data_offsets":[12,24]}
	}`, 24)

	headerSize, tensors, metadata, err := ReadSafetensorsHeader(bytes.NewReader(data), uint64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if headerSize == 0 || metadata["format"] != "pt" {
		t.Fatalf("unexpected metadata: header=%d metadata=%#v", headerSize, metadata)
	}
	if len(tensors) != 2 {
		t.Fatalf("unexpected tensor count: %#v", tensors)
	}
	if tensors[1].Name != "model.embed_tokens.weight" || tensors[1].DType != "F16" || tensors[1].Parameters != 6 || tensors[1].ByteSize != 12 {
		t.Fatalf("unexpected tensor: %#v", tensors[1])
	}
}

func TestInspectSafetensorsDirectory(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), safetensorsFixture(t, `{
		"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}
	}`, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := Inspect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.Safetensors == nil {
		t.Fatalf("missing safetensors info: %#v", info)
	}
	if info.Safetensors.ParamCount != 4 || info.Safetensors.DTypeCounts["F16"] != 1 {
		t.Fatalf("unexpected safetensors info: %#v", info.Safetensors)
	}
}

func TestReadSafetensorsHeaderRejectsInvalidSize(t *testing.T) {
	data := safetensorsFixture(t, `{
		"weight":{"dtype":"F16","shape":[2],"data_offsets":[0,3]}
	}`, 3)
	_, _, _, err := ReadSafetensorsHeader(bytes.NewReader(data), uint64(len(data)))
	if err == nil || !strings.Contains(err.Error(), "byte size") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func safetensorsFixture(t *testing.T, header string, dataBytes int) []byte {
	t.Helper()
	header = strings.TrimSpace(header)
	buf := make([]byte, 8+len(header)+dataBytes)
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(header)))
	copy(buf[8:], header)
	return buf
}
