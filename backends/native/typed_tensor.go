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
	case 2:
		return dequantizeQ4_0(data, elements)
	case 3:
		return dequantizeQ4_1(data, elements)
	case 6:
		return dequantizeQ5_0(data, elements)
	case 7:
		return dequantizeQ5_1(data, elements)
	case 8:
		return dequantizeQ8_0(data, elements)
	case 9:
		return dequantizeQ8_1(data, elements)
	case 10:
		return dequantizeQ2_K(data, elements)
	case 11:
		return dequantizeQ3_K(data, elements)
	case 12:
		return dequantizeQ4_K(data, elements)
	case 13:
		return dequantizeQ5_K(data, elements)
	case 14:
		return dequantizeQ6_K(data, elements)
	case 30:
		return dequantizeBF16(data, elements)
	default:
		return nil, fmt.Errorf("tensor %q type %s cannot be loaded as float32 yet", tensor.Name, tensor.Type)
	}
}

func cloneFloat32(values []float32) []float32 {
	if values == nil {
		return nil
	}
	out := make([]float32, len(values))
	copy(out, values)
	return out
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
