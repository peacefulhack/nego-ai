package native

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

func tensorFloat32(tensor modelinfo.GGUFTensor, data []byte) ([]float32, error) {
	elements, err := tensorElementCount(tensor)
	if err != nil {
		return nil, err
	}
	switch tensor.GGMLType {
	case 0:
		return dequantizeF32(data, elements)
	case 1:
		return dequantizeF16(data, elements)
	case 8:
		return dequantizeQ8_0(data, elements)
	case 30:
		return dequantizeBF16(data, elements)
	default:
		return nil, fmt.Errorf("tensor %q type %s cannot be loaded as float32 yet", tensor.Name, tensor.Type)
	}
}

func tensorElementCount(tensor modelinfo.GGUFTensor) (int, error) {
	if tensor.ElementCount == 0 {
		return 0, fmt.Errorf("tensor %q has no elements", tensor.Name)
	}
	if tensor.ElementCount > uint64(int(^uint(0)>>1)) {
		return 0, fmt.Errorf("tensor %q element count overflows this runtime", tensor.Name)
	}
	return int(tensor.ElementCount), nil
}
