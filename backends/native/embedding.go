package native

import (
	"fmt"

	"github.com/gakon/nego-ai/modelinfo"
)

func embeddingLookupFloat32(values []float32, tensor modelinfo.GGUFTensor, tokenID int, embeddingLength uint64) ([]float32, error) {
	dim, vocab, err := matrixShape(tensor, embeddingLength)
	if err != nil {
		return nil, err
	}
	if tokenID < 0 || tokenID >= vocab {
		return nil, fmt.Errorf("token id %d is out of embedding range 0..%d", tokenID, vocab-1)
	}
	if len(values) != dim*vocab {
		return nil, fmt.Errorf("embedding tensor %q has %d values, want %d", tensor.Name, len(values), dim*vocab)
	}
	start := tokenID * dim
	out := make([]float32, dim)
	copy(out, values[start:start+dim])
	return out, nil
}

func logitsFromOutputWeightFloat32(hidden, values []float32, tensor modelinfo.GGUFTensor) ([]float32, error) {
	if len(hidden) == 0 {
		return nil, fmt.Errorf("hidden state is empty")
	}
	dim, vocab, err := matrixShape(tensor, uint64(len(hidden)))
	if err != nil {
		return nil, err
	}
	if len(values) != dim*vocab {
		return nil, fmt.Errorf("output tensor %q has %d values, want %d", tensor.Name, len(values), dim*vocab)
	}
	logits := make([]float32, vocab)
	for token := 0; token < vocab; token++ {
		start := token * dim
		sum, err := dotFloat32(hidden, values[start:start+dim])
		if err != nil {
			return nil, err
		}
		logits[token] = sum
	}
	return logits, nil
}

func matrixShape(tensor modelinfo.GGUFTensor, expectedDim uint64) (int, int, error) {
	if len(tensor.Shape) != 2 {
		return 0, 0, fmt.Errorf("tensor %q must be 2D, got shape %v", tensor.Name, tensor.Shape)
	}
	if tensor.Shape[0] != expectedDim {
		return 0, 0, fmt.Errorf("tensor %q dim %d does not match expected dim %d", tensor.Name, tensor.Shape[0], expectedDim)
	}
	if tensor.Shape[1] == 0 {
		return 0, 0, fmt.Errorf("tensor %q vocab dimension is empty", tensor.Name)
	}
	if tensor.Shape[0] > uint64(int(^uint(0)>>1)) || tensor.Shape[1] > uint64(int(^uint(0)>>1)) {
		return 0, 0, fmt.Errorf("tensor %q shape overflows this runtime", tensor.Name)
	}
	dim := int(tensor.Shape[0])
	vocab := int(tensor.Shape[1])
	if dim != 0 && vocab > int(^uint(0)>>1)/dim {
		return 0, 0, fmt.Errorf("tensor %q value count overflows this runtime", tensor.Name)
	}
	return dim, vocab, nil
}
