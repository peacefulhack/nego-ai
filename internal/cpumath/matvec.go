// Package cpumath provides shared pure-Go CPU kernels for local runtimes.
package cpumath

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// MatVec multiplies a row-major matrix by vector without modifying either input.
// Large matrices use disjoint row ranges, preserving serial float32 accumulation
// within each row. Callers must not mutate inputs until MatVec returns.
func MatVec(matrix []float32, rows, cols int, vector []float32) ([]float32, error) {
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
	for _, value := range vector {
		if !finite(value) {
			return nil, fmt.Errorf("matrix vector contains non-finite value")
		}
	}
	out := make([]float32, rows)
	// Bound scheduling costs for small matrices and avoid excessive workers on
	// hosts with many logical CPUs. No cross-row reduction or shared scratch.
	workers := min(runtime.GOMAXPROCS(0), 8, rows, len(matrix)/(256*1024))
	if workers < 2 {
		if err := matVecRows(out, matrix, vector, cols, 0, rows); err != nil {
			return nil, err
		}
		return out, nil
	}
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		// Quotient/remainder splitting avoids overflow in rows*worker.
		start := worker*(rows/workers) + min(worker, rows%workers)
		end := start + rows/workers
		if worker < rows%workers {
			end++
		}
		if worker == workers-1 {
			errs[worker] = matVecRows(out, matrix, vector, cols, start, end)
			break
		}
		wg.Add(1)
		go func(worker, start, end int) {
			defer wg.Done()
			errs[worker] = matVecRows(out, matrix, vector, cols, start, end)
		}(worker, start, end)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func matVecRows(out, matrix, vector []float32, cols, start, end int) error {
	for row := start; row < end; row++ {
		weights := matrix[row*cols : (row+1)*cols]
		var sum float32
		for col, weight := range weights {
			if !finite(weight) {
				return fmt.Errorf("matrix row %d contains non-finite value", row)
			}
			sum += weight * vector[col]
		}
		if !finite(sum) {
			return fmt.Errorf("matrix row %d result is non-finite", row)
		}
		out[row] = sum
	}
	return nil
}

func finite(value float32) bool {
	return math.Float32bits(value)&0x7f800000 != 0x7f800000
}
