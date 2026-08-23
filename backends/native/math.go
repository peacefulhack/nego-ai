package native

import (
	"encoding/binary"
	"fmt"
	"math"
)

func dotFloat32(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("dot length mismatch: %d != %d", len(a), len(b))
	}
	var sum float32
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
			return 0, fmt.Errorf("dot input contains non-finite value")
		}
		sum += a[i] * b[i]
	}
	return sum, nil
}

func matVecFloat32(matrix []float32, rows, cols int, vector []float32) ([]float32, error) {
	if rows < 0 || cols < 0 {
		return nil, fmt.Errorf("matrix dimensions must be non-negative")
	}
	if len(vector) != cols {
		return nil, fmt.Errorf("vector length mismatch: %d != %d", len(vector), cols)
	}
	if rows != 0 && cols > int(^uint(0)>>1)/rows {
		return nil, fmt.Errorf("matrix dimensions overflow")
	}
	if len(matrix) != rows*cols {
		return nil, fmt.Errorf("matrix length mismatch: got %d, want %d", len(matrix), rows*cols)
	}
	out := make([]float32, rows)
	for row := 0; row < rows; row++ {
		start := row * cols
		sum, err := dotFloat32(matrix[start:start+cols], vector)
		if err != nil {
			return nil, err
		}
		out[row] = sum
	}
	return out, nil
}

func dequantizeF32(src []byte, elements int) ([]float32, error) {
	need, err := requiredBytes(elements, 4, "f32")
	if err != nil {
		return nil, err
	}
	if len(src) < need {
		return nil, fmt.Errorf("f32 tensor data is %d bytes, need %d", len(src), need)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
	return out, nil
}

func dequantizeF16(src []byte, elements int) ([]float32, error) {
	need, err := requiredBytes(elements, 2, "f16")
	if err != nil {
		return nil, err
	}
	if len(src) < need {
		return nil, fmt.Errorf("f16 tensor data is %d bytes, need %d", len(src), need)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = float16ToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
	}
	return out, nil
}

func dequantizeBF16(src []byte, elements int) ([]float32, error) {
	need, err := requiredBytes(elements, 2, "bf16")
	if err != nil {
		return nil, err
	}
	if len(src) < need {
		return nil, fmt.Errorf("bf16 tensor data is %d bytes, need %d", len(src), need)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = bf16ToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
	}
	return out, nil
}

func requiredBytes(elements, bytesPerElement int, name string) (int, error) {
	if elements < 0 {
		return 0, fmt.Errorf("%s element count must be non-negative", name)
	}
	if bytesPerElement <= 0 {
		return 0, fmt.Errorf("%s bytes per element must be positive", name)
	}
	if elements > int(^uint(0)>>1)/bytesPerElement {
		return 0, fmt.Errorf("%s tensor byte size overflows", name)
	}
	return elements * bytesPerElement, nil
}

func dequantizeQ8_0(src []byte, elements int) ([]float32, error) {
	if elements < 0 {
		return nil, fmt.Errorf("element count must be non-negative")
	}
	layout := ggmlLayouts[8]
	blocks := (uint64(elements) + layout.blockSize - 1) / layout.blockSize
	need := blocks * layout.typeSize
	if uint64(len(src)) < need {
		return nil, fmt.Errorf("q8_0 tensor data is %d bytes, need %d", len(src), need)
	}
	out := make([]float32, elements)
	for block := uint64(0); block < blocks; block++ {
		start := block * layout.typeSize
		scale := float16ToFloat32(binary.LittleEndian.Uint16(src[start:]))
		values := src[start+2 : start+layout.typeSize]
		for i := uint64(0); i < layout.blockSize; i++ {
			outIndex := block*layout.blockSize + i
			if outIndex >= uint64(elements) {
				break
			}
			out[outIndex] = scale * float32(int8(values[i]))
		}
	}
	return out, nil
}

func float16ToFloat32(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exp := (value >> 10) & 0x1f
	frac := uint32(value & 0x03ff)

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		e := int32(-14)
		for frac&0x0400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x03ff
		return math.Float32frombits(sign | uint32(e+127)<<23 | frac<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	default:
		return math.Float32frombits(sign | uint32(exp+112)<<23 | frac<<13)
	}
}

func bf16ToFloat32(value uint16) float32 {
	return math.Float32frombits(uint32(value) << 16)
}
