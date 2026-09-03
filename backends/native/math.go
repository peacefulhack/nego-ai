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

func dequantizeQ4_0(src []byte, elements int) ([]float32, error) {
	return dequantizeQ4(src, elements, 2, 0, false, "q4_0")
}

func dequantizeQ4_1(src []byte, elements int) ([]float32, error) {
	return dequantizeQ4(src, elements, 3, 2, true, "q4_1")
}

func dequantizeQ4(src []byte, elements int, typeID uint32, minOffset uint64, unsigned bool, name string) ([]float32, error) {
	if elements < 0 {
		return nil, fmt.Errorf("element count must be non-negative")
	}
	layout := ggmlLayouts[typeID]
	blocks := (uint64(elements) + layout.blockSize - 1) / layout.blockSize
	need := blocks * layout.typeSize
	if uint64(len(src)) < need {
		return nil, fmt.Errorf("%s tensor data is %d bytes, need %d", name, len(src), need)
	}
	out := make([]float32, elements)
	for block := uint64(0); block < blocks; block++ {
		start := block * layout.typeSize
		scale := float16ToFloat32(binary.LittleEndian.Uint16(src[start:]))
		minValue := float32(0)
		if minOffset != 0 {
			minValue = float16ToFloat32(binary.LittleEndian.Uint16(src[start+minOffset:]))
		}
		quantized := src[start+layout.typeSize-16 : start+layout.typeSize]
		writeQ4Block(out, block, layout.blockSize, scale, minValue, quantized, unsigned)
	}
	return out, nil
}

func writeQ4Block(out []float32, block, blockSize uint64, scale, minValue float32, quantized []byte, unsigned bool) {
	half := blockSize / 2
	for i := uint64(0); i < half; i++ {
		packed := quantized[i]
		writeQ4Value(out, block*blockSize+i, scale, minValue, packed&0x0f, unsigned)
		writeQ4Value(out, block*blockSize+half+i, scale, minValue, packed>>4, unsigned)
	}
}

func writeQ4Value(out []float32, index uint64, scale, minValue float32, value byte, unsigned bool) {
	if index >= uint64(len(out)) {
		return
	}
	quantized := int(value)
	if !unsigned {
		quantized -= 8
	}
	out[index] = scale*float32(quantized) + minValue
}

func dequantizeQ5_0(src []byte, elements int) ([]float32, error) {
	return dequantizeQ5(src, elements, 6, 2, false, "q5_0")
}

func dequantizeQ5_1(src []byte, elements int) ([]float32, error) {
	return dequantizeQ5(src, elements, 7, 4, true, "q5_1")
}

func dequantizeQ5(src []byte, elements int, typeID uint32, qhOffset uint64, unsigned bool, name string) ([]float32, error) {
	if elements < 0 {
		return nil, fmt.Errorf("element count must be non-negative")
	}
	layout := ggmlLayouts[typeID]
	blocks := (uint64(elements) + layout.blockSize - 1) / layout.blockSize
	need := blocks * layout.typeSize
	if uint64(len(src)) < need {
		return nil, fmt.Errorf("%s tensor data is %d bytes, need %d", name, len(src), need)
	}
	out := make([]float32, elements)
	for block := uint64(0); block < blocks; block++ {
		start := block * layout.typeSize
		scale := float16ToFloat32(binary.LittleEndian.Uint16(src[start:]))
		minValue := float32(0)
		if unsigned {
			minValue = float16ToFloat32(binary.LittleEndian.Uint16(src[start+2:]))
		}
		highBits := binary.LittleEndian.Uint32(src[start+qhOffset:])
		quantized := src[start+layout.typeSize-16 : start+layout.typeSize]
		writeQ5Block(out, block, layout.blockSize, scale, minValue, highBits, quantized, unsigned)
	}
	return out, nil
}

func writeQ5Block(out []float32, block, blockSize uint64, scale, minValue float32, highBits uint32, quantized []byte, unsigned bool) {
	half := blockSize / 2
	for i := uint64(0); i < half; i++ {
		packed := quantized[i]
		lowIndex := block*blockSize + i
		highIndex := block*blockSize + half + i
		writeQ5Value(out, lowIndex, scale, minValue, packed&0x0f, highBits, i, unsigned)
		writeQ5Value(out, highIndex, scale, minValue, packed>>4, highBits, half+i, unsigned)
	}
}

func writeQ5Value(out []float32, index uint64, scale, minValue float32, lowBits byte, highBits uint32, bitIndex uint64, unsigned bool) {
	if index >= uint64(len(out)) {
		return
	}
	quantized := int(lowBits)
	if (highBits>>bitIndex)&1 == 1 {
		quantized |= 16
	}
	if !unsigned {
		quantized -= 16
	}
	out[index] = scale*float32(quantized) + minValue
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

func dequantizeQ8_1(src []byte, elements int) ([]float32, error) {
	if elements < 0 {
		return nil, fmt.Errorf("element count must be non-negative")
	}
	layout := ggmlLayouts[9]
	blocks := (uint64(elements) + layout.blockSize - 1) / layout.blockSize
	need := blocks * layout.typeSize
	if uint64(len(src)) < need {
		return nil, fmt.Errorf("q8_1 tensor data is %d bytes, need %d", len(src), need)
	}
	out := make([]float32, elements)
	for block := uint64(0); block < blocks; block++ {
		start := block * layout.typeSize
		scale := float16ToFloat32(binary.LittleEndian.Uint16(src[start:]))
		values := src[start+4 : start+layout.typeSize]
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

func dequantizeQ4_K(src []byte, elements int) ([]float32, error) {
	return dequantizeQKWithMin(src, elements, 12, "q4_k", false)
}

func dequantizeQ5_K(src []byte, elements int) ([]float32, error) {
	return dequantizeQKWithMin(src, elements, 13, "q5_k", true)
}

func dequantizeQKWithMin(src []byte, elements int, typeID uint32, name string, highBits bool) ([]float32, error) {
	if elements < 0 {
		return nil, fmt.Errorf("element count must be non-negative")
	}
	layout := ggmlLayouts[typeID]
	blocks := (uint64(elements) + layout.blockSize - 1) / layout.blockSize
	need := blocks * layout.typeSize
	if uint64(len(src)) < need {
		return nil, fmt.Errorf("%s tensor data is %d bytes, need %d", name, len(src), need)
	}
	out := make([]float32, elements)
	for block := uint64(0); block < blocks; block++ {
		start := block * layout.typeSize
		scale := float16ToFloat32(binary.LittleEndian.Uint16(src[start:]))
		minScale := float16ToFloat32(binary.LittleEndian.Uint16(src[start+2:]))
		scales := src[start+4 : start+16]
		qhOffset := start + 16
		qsOffset := qhOffset
		if highBits {
			qsOffset += 32
		}
		quantized := src[qsOffset : qsOffset+128]
		for group := 0; group < 8; group++ {
			groupScale, groupMin := unpackKScaleMin(scales, group)
			d := scale * float32(groupScale)
			m := minScale * float32(groupMin)
			for i := 0; i < 32; i++ {
				outIndex := block*layout.blockSize + uint64(group*32+i)
				if outIndex >= uint64(len(out)) {
					return out, nil
				}
				packed := quantized[(group/2)*32+i]
				q := int(packed & 0x0f)
				if group%2 == 1 {
					q = int(packed >> 4)
				}
				if highBits && ((src[qhOffset+uint64(i)]>>uint(group))&1) != 0 {
					q |= 16
				}
				out[outIndex] = d*float32(q) - m
			}
		}
	}
	return out, nil
}

func dequantizeQ6_K(src []byte, elements int) ([]float32, error) {
	if elements < 0 {
		return nil, fmt.Errorf("element count must be non-negative")
	}
	layout := ggmlLayouts[14]
	blocks := (uint64(elements) + layout.blockSize - 1) / layout.blockSize
	need := blocks * layout.typeSize
	if uint64(len(src)) < need {
		return nil, fmt.Errorf("q6_k tensor data is %d bytes, need %d", len(src), need)
	}
	out := make([]float32, elements)
	for block := uint64(0); block < blocks; block++ {
		start := block * layout.typeSize
		ql := src[start : start+128]
		qh := src[start+128 : start+192]
		scales := src[start+192 : start+208]
		scale := float16ToFloat32(binary.LittleEndian.Uint16(src[start+208:]))
		for group := 0; group < 16; group++ {
			groupScale := scale * float32(int8(scales[group]))
			for i := 0; i < 16; i++ {
				valueIndex := group*16 + i
				outIndex := block*layout.blockSize + uint64(valueIndex)
				if outIndex >= uint64(len(out)) {
					return out, nil
				}
				q := q6KValue(ql, qh, valueIndex)
				out[outIndex] = groupScale * float32(q)
			}
		}
	}
	return out, nil
}

func unpackKScaleMin(scales []byte, group int) (int, int) {
	if group < 4 {
		return int(scales[group] & 0x3f), int(scales[group+4] & 0x3f)
	}
	scale := int((scales[group+4] & 0x0f) | ((scales[group-4] >> 6) << 4))
	minValue := int((scales[group+4] >> 4) | ((scales[group] >> 6) << 4))
	return scale, minValue
}

func q6KValue(ql, qh []byte, index int) int {
	qGroup := index / 32
	offset := index % 32

	lowChunk := qGroup / 4
	lowGroup := qGroup % 4
	lowOffset := lowChunk*64 + offset
	if lowGroup%2 == 1 {
		lowOffset += 32
	}
	lowShift := uint(0)
	if lowGroup >= 2 {
		lowShift = 4
	}
	low := int((ql[lowOffset] >> lowShift) & 0x0f)

	highChunk := qGroup / 4
	highShift := uint((qGroup % 4) * 2)
	highByte := qh[highChunk*32+offset]
	high := int((highByte >> highShift) & 0x03)
	return (low | (high << 4)) - 32
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
