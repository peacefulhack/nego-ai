package native

import (
	"context"
	"math"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

func TestLoadTensorFloat32(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t), Options: allowIncompleteOptions()})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel := model.(*Model)
	values, tensor, err := nativeModel.LoadTensorFloat32("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	if tensor.Name != "token_embd.weight" {
		t.Fatalf("unexpected tensor: %s", tensor.Name)
	}
	if len(values) != 128 {
		t.Fatalf("unexpected value count: %d", len(values))
	}
	stats, ok := nego.RuntimeStatsOf(model)
	if !ok {
		t.Fatal("expected native model to expose runtime stats")
	}
	if stats.Backend != BackendName || stats.Device != "cpu" || !stats.TensorCacheEnabled || stats.CachedTensors != 1 || stats.CachedTensorBytes != 512 {
		t.Fatalf("unexpected runtime stats: %#v", stats)
	}
}

func TestLoadTensorFloat32ReturnsCopyFromCache(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t), Options: allowIncompleteOptions()})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel := model.(*Model)
	first, _, err := nativeModel.LoadTensorFloat32("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	first[0] = 42
	second, _, err := nativeModel.LoadTensorFloat32("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	if second[0] == 42 {
		t.Fatal("LoadTensorFloat32 returned mutable cached data")
	}
}

func TestLoadTensorFloat32SharedUsesCache(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t), Options: allowIncompleteOptions()})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel := model.(*Model)
	first, _, err := nativeModel.loadTensorFloat32Shared("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	second, _, err := nativeModel.loadTensorFloat32Shared("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(first) == 0 || len(second) == 0 || &first[0] != &second[0] {
		t.Fatal("expected shared tensor cache to reuse the same float32 buffer")
	}
}

func TestLoadTensorFloat32CanDisableCache(t *testing.T) {
	options := allowIncompleteOptions()
	options["cache_tensors"] = "false"
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t), Options: options})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel := model.(*Model)
	first, _, err := nativeModel.loadTensorFloat32Shared("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	second, _, err := nativeModel.loadTensorFloat32Shared("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(first) == 0 || len(second) == 0 {
		t.Fatal("expected decoded tensors")
	}
	if &first[0] == &second[0] {
		t.Fatal("expected cache_tensors=false to avoid reusing the same float32 buffer")
	}
	stats, ok := nego.RuntimeStatsOf(model)
	if !ok {
		t.Fatal("expected native model to expose runtime stats")
	}
	if stats.TensorCacheEnabled || stats.CachedTensors != 0 || stats.CachedTensorBytes != 0 {
		t.Fatalf("unexpected no-cache runtime stats: %#v", stats)
	}
}

func TestLoadTensorFloat32HonorsCacheByteLimit(t *testing.T) {
	options := allowIncompleteOptions()
	options["max_tensor_cache_bytes"] = "1"
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t), Options: options})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	nativeModel := model.(*Model)
	_, _, err = nativeModel.loadTensorFloat32Shared("token_embd.weight")
	if err == nil || !strings.Contains(err.Error(), "max_tensor_cache_bytes") {
		t.Fatalf("unexpected cache limit error: %v", err)
	}
}

func TestLoadTensorFloat32RejectsClosedModel(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t), Options: allowIncompleteOptions()})
	if err != nil {
		t.Fatal(err)
	}
	nativeModel := model.(*Model)
	if err := nativeModel.Close(); err != nil {
		t.Fatal(err)
	}
	_, _, err = nativeModel.LoadTensorFloat32("token_embd.weight")
	if err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTensorFloat32RejectsUnsupportedType(t *testing.T) {
	_, err := tensorFloat32(modelinfo.GGUFTensor{
		Name:         "blk.0.attn_q.weight",
		Type:         "i8",
		GGMLType:     24,
		ElementCount: 1,
	}, make([]byte, 1))
	if err == nil || !strings.Contains(err.Error(), "cannot be loaded as float32 yet") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestQuantizedTensorLayoutsLoadFloat32(t *testing.T) {
	cases := []struct {
		name string
		typ  uint32
	}{
		{name: "q4_0", typ: 2},
		{name: "q4_1", typ: 3},
		{name: "q5_0", typ: 6},
		{name: "q5_1", typ: 7},
		{name: "q8_0", typ: 8},
		{name: "q8_1", typ: 9},
		{name: "q2_k", typ: 10},
		{name: "q3_k", typ: 11},
		{name: "q4_k", typ: 12},
		{name: "q5_k", typ: 13},
		{name: "q6_k", typ: 14},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			layout := ggmlLayouts[tc.typ]
			tensor := modelinfo.GGUFTensor{
				Name:         tc.name + ".weight",
				Type:         tc.name,
				GGMLType:     tc.typ,
				ElementCount: layout.blockSize,
			}
			size, err := tensorByteSize(tensor)
			if err != nil {
				t.Fatal(err)
			}
			if size != layout.typeSize {
				t.Fatalf("tensorByteSize = %d, want %d", size, layout.typeSize)
			}
			values, err := tensorFloat32(tensor, make([]byte, layout.typeSize))
			if err != nil {
				t.Fatal(err)
			}
			if len(values) != int(layout.blockSize) {
				t.Fatalf("decoded values = %d, want %d", len(values), layout.blockSize)
			}
			for i, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("decoded value %d is non-finite: %v", i, value)
				}
			}
		})
	}
}

func TestQuantizedTensorByteSizeRoundsToBlocks(t *testing.T) {
	layout := ggmlLayouts[12]
	size, err := tensorByteSize(modelinfo.GGUFTensor{
		Name:         "partial-q4-k.weight",
		Type:         "q4_k",
		GGMLType:     12,
		ElementCount: layout.blockSize + 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if size != layout.typeSize*2 {
		t.Fatalf("tensorByteSize = %d, want %d", size, layout.typeSize*2)
	}
}

func TestTensorElementCountOverflow(t *testing.T) {
	_, err := tensorElementCount(modelinfo.GGUFTensor{
		Name:         "huge.weight",
		ElementCount: uint64(int(^uint(0)>>1)) + 1,
	})
	if err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("unexpected error: %v", err)
	}
}
