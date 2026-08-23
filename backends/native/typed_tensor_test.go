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

func TestTensorFloat32RejectsUnsupportedType(t *testing.T) {
	_, err := tensorFloat32(modelinfo.GGUFTensor{
		Name:         "blk.0.attn_q.weight",
		Type:         "q4_0",
		GGMLType:     2,
		ElementCount: 32,
	}, make([]byte, 18))
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
