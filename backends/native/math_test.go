package native

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestFloat16ToFloat32(t *testing.T) {
	cases := []struct {
		name string
		in   uint16
		want float32
	}{
		{name: "zero", in: 0x0000, want: 0},
		{name: "one", in: 0x3c00, want: 1},
		{name: "negative two", in: 0xc000, want: -2},
		{name: "half", in: 0x3800, want: 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := float16ToFloat32(tc.in); got != tc.want {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
	if !math.IsInf(float64(float16ToFloat32(0x7c00)), 1) {
		t.Fatal("expected +Inf")
	}
	if !math.IsNaN(float64(float16ToFloat32(0x7e00))) {
		t.Fatal("expected NaN")
	}
	if got := float16ToFloat32(0x0001); got <= 0 {
		t.Fatalf("expected positive subnormal, got %v", got)
	}
}

func TestDequantizeFloatTypes(t *testing.T) {
	var f32 [8]byte
	binary.LittleEndian.PutUint32(f32[0:], math.Float32bits(1.25))
	binary.LittleEndian.PutUint32(f32[4:], math.Float32bits(-2.5))
	gotF32, err := dequantizeF32(f32[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, gotF32, []float32{1.25, -2.5})

	f16 := []byte{0x00, 0x3c, 0x00, 0xc0}
	gotF16, err := dequantizeF16(f16, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, gotF16, []float32{1, -2})

	bf16 := []byte{0x80, 0x3f, 0x20, 0xc0}
	gotBF16, err := dequantizeBF16(bf16, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, gotBF16, []float32{1, -2.5})
}

func TestDequantizeQ8_0(t *testing.T) {
	block := make([]byte, 34)
	binary.LittleEndian.PutUint16(block[0:], 0x3c00)
	copy(block[2:], []byte{254, 255, 0, 1})
	got, err := dequantizeQ8_0(block, 4)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{-2, -1, 0, 1})
}

func TestDequantizeQ4_0(t *testing.T) {
	block := make([]byte, 18)
	binary.LittleEndian.PutUint16(block[0:], 0x3c00)
	block[2] = 0x8f
	got, err := dequantizeQ4_0(block, 17)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got[:2], []float32{7, -8})
	if got[16] != 0 {
		t.Fatalf("unexpected high nibble value: %v", got[16])
	}
}

func TestDequantizeQ4_1(t *testing.T) {
	block := make([]byte, 20)
	binary.LittleEndian.PutUint16(block[0:], 0x3c00)
	binary.LittleEndian.PutUint16(block[2:], 0x4000)
	block[4] = 0x21
	got, err := dequantizeQ4_1(block, 17)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got[:2], []float32{3, 2})
	if got[16] != 4 {
		t.Fatalf("unexpected high nibble value: %v", got[16])
	}
}

func TestDequantizeQ5(t *testing.T) {
	q5_0 := make([]byte, 22)
	binary.LittleEndian.PutUint16(q5_0[0:], 0x3c00)
	binary.LittleEndian.PutUint32(q5_0[2:], 1)
	q5_0[6] = 0x0f
	gotQ50, err := dequantizeQ5_0(q5_0, 17)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, gotQ50[:2], []float32{15, -16})
	if gotQ50[16] != -16 {
		t.Fatalf("unexpected high nibble value: %v", gotQ50[16])
	}

	q5_1 := make([]byte, 24)
	binary.LittleEndian.PutUint16(q5_1[0:], 0x3c00)
	binary.LittleEndian.PutUint16(q5_1[2:], 0x4000)
	binary.LittleEndian.PutUint32(q5_1[4:], 1)
	q5_1[8] = 0x0f
	gotQ51, err := dequantizeQ5_1(q5_1, 17)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, gotQ51[:2], []float32{33, 2})
	if gotQ51[16] != 2 {
		t.Fatalf("unexpected high nibble value: %v", gotQ51[16])
	}
}

func TestDequantizeQ8_1(t *testing.T) {
	block := make([]byte, 40)
	binary.LittleEndian.PutUint16(block[0:], 0x3c00)
	copy(block[4:], []byte{254, 255, 0, 1})
	got, err := dequantizeQ8_1(block, 4)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{-2, -1, 0, 1})
}

func TestMatVecFloat32(t *testing.T) {
	got, err := matVecFloat32([]float32{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3, []float32{1, 0.5, -1})
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Slice(t, got, []float32{-1, 0.5})
}

func TestDotFloat32RejectsInvalidInputs(t *testing.T) {
	if _, err := dotFloat32([]float32{float32(math.NaN())}, []float32{1}); err == nil {
		t.Fatal("expected invalid dot input error")
	}
}

func assertFloat32Slice(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("index %d: got %v, want %v", i, got[i], want[i])
		}
	}
}
