package cpumath

import (
	"fmt"
	"math"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func TestMatVecMatchesSerial(t *testing.T) {
	for _, shape := range [][2]int{{0, 0}, {3, 0}, {0, 3}, {3, 5}, {1031, 1024}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			rows, cols := shape[0], shape[1]
			matrix, vector := matrixFixture(rows, cols)
			matrixCopy, vectorCopy := append([]float32{}, matrix...), append([]float32{}, vector...)
			want, err := serialReference(matrix, rows, cols, vector)
			if err != nil {
				t.Fatal(err)
			}
			for _, threads := range []int{1, 2, 8} {
				previous := runtime.GOMAXPROCS(threads)
				got, err := MatVec(matrix, rows, cols, vector)
				runtime.GOMAXPROCS(previous)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("threads=%d: result differs from serial reference; error=%v", threads, err)
				}
			}
			if !reflect.DeepEqual(matrix, matrixCopy) || !reflect.DeepEqual(vector, vectorCopy) {
				t.Fatal("MatVec mutated its inputs")
			}
		})
	}
}

func TestMatVecConcurrentCalls(t *testing.T) {
	matrix, vector := matrixFixture(1031, 1024)
	want, err := serialReference(matrix, 1031, 1024, vector)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := MatVec(matrix, 1031, 1024, vector)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Errorf("concurrent result differs; error=%v", err)
			}
		}()
	}
	wg.Wait()
}

func TestMatVecRejectsInvalidInput(t *testing.T) {
	for _, tc := range []struct {
		name       string
		matrix     []float32
		rows, cols int
		vector     []float32
		wantError  string
	}{
		{"negative rows", nil, -1, 0, nil, "non-negative"},
		{"negative cols", nil, 0, -1, nil, "non-negative"},
		{"dimension overflow", nil, int(^uint(0) >> 1), 2, []float32{1, 1}, "overflow"},
		{"wrong matrix", []float32{1}, 2, 2, []float32{1, 1}, "matrix length mismatch"},
		{"wrong vector", []float32{1}, 1, 1, nil, "vector length mismatch"},
		{"nan matrix", []float32{float32(math.NaN())}, 1, 1, []float32{1}, "non-finite"},
		{"inf matrix", []float32{float32(math.Inf(-1))}, 1, 1, []float32{1}, "non-finite"},
		{"nan vector", []float32{1}, 1, 1, []float32{float32(math.NaN())}, "non-finite"},
		{"inf vector", []float32{1}, 1, 1, []float32{float32(math.Inf(1))}, "non-finite"},
		{"product overflow", []float32{math.MaxFloat32}, 1, 1, []float32{2}, "result is non-finite"},
		{"sum overflow", []float32{math.MaxFloat32, math.MaxFloat32}, 1, 2, []float32{1, 1}, "result is non-finite"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := MatVec(tc.matrix, tc.rows, tc.cols, tc.vector)
			if err == nil || !strings.Contains(err.Error(), tc.wantError) || out != nil {
				t.Fatalf("out=%v error=%v, want %q", out, err, tc.wantError)
			}
		})
	}
}

func TestMatVecParallelError(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	for _, badRow := range []int{0, 520, 1030} {
		matrix, vector := matrixFixture(1031, 1024)
		matrix[badRow*1024+1023] = float32(math.NaN())
		out, err := MatVec(matrix, 1031, 1024, vector)
		if out != nil || err == nil || !strings.Contains(err.Error(), fmt.Sprintf("row %d", badRow)) {
			t.Fatalf("row %d: expected no partial output and an error, got %v", badRow, err)
		}
	}
}

func TestMatVecAcceptsSubnormal(t *testing.T) {
	out, err := MatVec([]float32{math.SmallestNonzeroFloat32}, 1, 1, []float32{1})
	if err != nil || len(out) != 1 || out[0] != math.SmallestNonzeroFloat32 {
		t.Fatalf("out=%v error=%v", out, err)
	}
}

func matrixFixture(rows, cols int) ([]float32, []float32) {
	matrix, vector := make([]float32, rows*cols), make([]float32, cols)
	for i := range matrix {
		matrix[i] = float32(i%29-14) / 31
	}
	for i := range vector {
		vector[i] = float32(i%17-8) / 19
	}
	return matrix, vector
}

// This retains the previous per-row dot-product checks as a benchmark baseline.
func serialReference(matrix []float32, rows, cols int, vector []float32) ([]float32, error) {
	out := make([]float32, rows)
	for row := range out {
		var sum float32
		for col, weight := range matrix[row*cols : (row+1)*cols] {
			value := vector[col]
			if math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite input")
			}
			sum += weight * value
		}
		out[row] = sum
	}
	return out, nil
}

var benchmarkOutput []float32

func BenchmarkMatVec(b *testing.B) {
	for _, shape := range [][2]int{{32, 32}, {1024, 1024}, {4096, 1024}} {
		b.Run(fmt.Sprintf("%dx%d", shape[0], shape[1]), func(b *testing.B) {
			matrix, vector := matrixFixture(shape[0], shape[1])
			for _, impl := range []struct {
				name string
				fn   func([]float32, int, int, []float32) ([]float32, error)
			}{{"reference", serialReference}, {"native", MatVec}} {
				b.Run(impl.name, func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						out, err := impl.fn(matrix, shape[0], shape[1], vector)
						if err != nil {
							b.Fatal(err)
						}
						benchmarkOutput = out
					}
				})
			}
		})
	}
}
