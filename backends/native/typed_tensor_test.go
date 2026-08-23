package native

import (
	"context"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

func TestLoadTensorFloat32(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
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
}

func TestLoadTensorFloat32ReturnsCopyFromCache(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
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
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
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

func TestLoadTensorFloat32RejectsClosedModel(t *testing.T) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeGGUF(t)})
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
		Type:         "q4_k",
		GGMLType:     12,
		ElementCount: 32,
	}, make([]byte, 144))
	if err == nil || !strings.Contains(err.Error(), "cannot be loaded as float32 yet") {
		t.Fatalf("unexpected error: %v", err)
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
