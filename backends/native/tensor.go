package native

import (
	"fmt"
	"io"
	"os"

	"github.com/gakon/nego-ai/modelinfo"
)

const maxTensorReadBytes = 512 << 20

type tensorStore struct {
	file     *os.File
	fileSize uint64
	tensors  map[string]modelinfo.GGUFTensor
}

func openTensorStore(path string, info *modelinfo.GGUFInfo) (*tensorStore, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}
	store := &tensorStore{
		file:     file,
		fileSize: uint64(stat.Size()),
		tensors:  make(map[string]modelinfo.GGUFTensor, len(info.Tensors)),
	}
	for _, tensor := range info.Tensors {
		store.tensors[tensor.Name] = tensor
	}
	return store, nil
}

func (s *tensorStore) Close() error {
	if s == nil || s.file == nil {
		return nil
	}
	return s.file.Close()
}

func (s *tensorStore) ReadTensor(name string) ([]byte, modelinfo.GGUFTensor, error) {
	reader, tensor, err := s.TensorReader(name)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	if reader.Size() > maxTensorReadBytes {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("tensor %q is %d bytes; use TensorReader for large tensors", name, reader.Size())
	}
	buf := make([]byte, reader.Size())
	if _, err := io.ReadFull(reader, buf); err != nil {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return buf, tensor, nil
}

func (s *tensorStore) TensorReader(name string) (*io.SectionReader, modelinfo.GGUFTensor, error) {
	tensor, ok := s.tensors[name]
	if !ok {
		return nil, modelinfo.GGUFTensor{}, fmt.Errorf("tensor %q not found", name)
	}
	size, err := tensorByteSize(tensor)
	if err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	if err := s.validateTensorBounds(tensor, size); err != nil {
		return nil, modelinfo.GGUFTensor{}, err
	}
	return io.NewSectionReader(s.file, int64(tensor.AbsoluteOffset), int64(size)), tensor, nil
}

func (s *tensorStore) validateTensorBounds(tensor modelinfo.GGUFTensor, size uint64) error {
	const maxInt64 = uint64(1<<63 - 1)
	if tensor.AbsoluteOffset > maxInt64 || size > maxInt64 {
		return fmt.Errorf("tensor %q cannot be addressed on this runtime", tensor.Name)
	}
	if tensor.AbsoluteOffset > s.fileSize || size > s.fileSize-tensor.AbsoluteOffset {
		return fmt.Errorf("tensor %q exceeds GGUF file bounds", tensor.Name)
	}
	return nil
}

func tensorByteSize(tensor modelinfo.GGUFTensor) (uint64, error) {
	layout, ok := ggmlLayouts[tensor.GGMLType]
	if !ok {
		return 0, fmt.Errorf("unsupported tensor type %s for %q", tensor.Type, tensor.Name)
	}
	if tensor.ElementCount == 0 {
		return 0, fmt.Errorf("tensor %q has no elements", tensor.Name)
	}
	blocks := tensor.ElementCount / layout.blockSize
	if tensor.ElementCount%layout.blockSize != 0 {
		blocks++
	}
	if blocks > ^uint64(0)/layout.typeSize {
		return 0, fmt.Errorf("tensor %q byte size overflows", tensor.Name)
	}
	return blocks * layout.typeSize, nil
}

type ggmlLayout struct {
	blockSize uint64
	typeSize  uint64
}

var ggmlLayouts = map[uint32]ggmlLayout{
	0:  {blockSize: 1, typeSize: 4},   // f32
	1:  {blockSize: 1, typeSize: 2},   // f16
	2:  {blockSize: 32, typeSize: 18}, // q4_0
	3:  {blockSize: 32, typeSize: 20}, // q4_1
	6:  {blockSize: 32, typeSize: 22}, // q5_0
	7:  {blockSize: 32, typeSize: 24}, // q5_1
	8:  {blockSize: 32, typeSize: 34}, // q8_0
	9:  {blockSize: 32, typeSize: 40}, // q8_1
	10: {blockSize: 256, typeSize: 84},
	11: {blockSize: 256, typeSize: 110},
	12: {blockSize: 256, typeSize: 144},
	13: {blockSize: 256, typeSize: 176},
	14: {blockSize: 256, typeSize: 210},
	24: {blockSize: 1, typeSize: 1}, // i8
	25: {blockSize: 1, typeSize: 2}, // i16
	26: {blockSize: 1, typeSize: 4}, // i32
	27: {blockSize: 1, typeSize: 8}, // i64
	28: {blockSize: 1, typeSize: 8}, // f64
	30: {blockSize: 1, typeSize: 2}, // bf16
}
