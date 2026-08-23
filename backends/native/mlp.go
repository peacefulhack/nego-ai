package native

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

func linearFloat32(input, values []float32, tensor modelinfo.GGUFTensor) ([]float32, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("linear input is empty")
	}
	in, out, err := matrixShape(tensor, uint64(len(input)))
	if err != nil {
		return nil, err
	}
	if len(values) != in*out {
		return nil, fmt.Errorf("linear tensor %q has %d values, want %d", tensor.Name, len(values), in*out)
	}
	result := make([]float32, out)
	for row := 0; row < out; row++ {
		start := row * in
		sum, err := dotFloat32(input, values[start:start+in])
		if err != nil {
			return nil, err
		}
		result[row] = sum
	}
	return result, nil
}

func mlpFloat32(input []float32, gateValues, upValues, downValues []float32, gateTensor, upTensor, downTensor modelinfo.GGUFTensor) ([]float32, error) {
	gate, err := linearFloat32(input, gateValues, gateTensor)
	if err != nil {
		return nil, fmt.Errorf("gate projection: %w", err)
	}
	up, err := linearFloat32(input, upValues, upTensor)
	if err != nil {
		return nil, fmt.Errorf("up projection: %w", err)
	}
	activated, err := siluFloat32(gate)
	if err != nil {
		return nil, err
	}
	hidden, err := mulFloat32(activated, up)
	if err != nil {
		return nil, err
	}
	out, err := linearFloat32(hidden, downValues, downTensor)
	if err != nil {
		return nil, fmt.Errorf("down projection: %w", err)
	}
	return out, nil
}
