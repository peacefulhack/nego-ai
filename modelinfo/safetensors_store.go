package modelinfo

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

const maxSafetensorsTensorReadBytes = 512 << 20

type SafetensorsStore struct {
	info    *SafetensorsInfo
	rootDir string
}

type SafetensorsTensorReader struct {
	*io.SectionReader
	file *os.File
}

func (r *SafetensorsTensorReader) Close() error {
	if r == nil || r.file == nil {
		return nil
	}
	return r.file.Close()
}

func OpenSafetensors(path string) (*SafetensorsStore, error) {
	info, err := InspectSafetensors(path)
	if err != nil {
		return nil, err
	}
	rootDir := info.Path
	if stat, err := os.Stat(info.Path); err == nil && !stat.IsDir() {
		rootDir = filepath.Dir(info.Path)
	}
	return &SafetensorsStore{info: info, rootDir: rootDir}, nil
}

func (s *SafetensorsStore) Info() *SafetensorsInfo {
	if s == nil {
		return nil
	}
	return s.info
}

func (s *SafetensorsStore) ReadTensor(name string) ([]byte, SafetensorsTensor, error) {
	tensor, err := s.tensorByName(name)
	if err != nil {
		return nil, SafetensorsTensor{}, err
	}
	if tensor.ByteSize > maxSafetensorsTensorReadBytes {
		return nil, SafetensorsTensor{}, fmt.Errorf("tensor %q is %d bytes; use TensorReader for large tensors", name, tensor.ByteSize)
	}
	reader, tensor, err := s.TensorReader(name)
	if err != nil {
		return nil, SafetensorsTensor{}, err
	}
	defer reader.Close()
	data := make([]byte, tensor.ByteSize)
	if _, err := io.ReadFull(reader, data); err != nil {
		return nil, SafetensorsTensor{}, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return data, tensor, nil
}

func (s *SafetensorsStore) TensorReader(name string) (*SafetensorsTensorReader, SafetensorsTensor, error) {
	tensor, headerSize, path, err := s.findTensor(name)
	if err != nil {
		return nil, SafetensorsTensor{}, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, SafetensorsTensor{}, err
	}
	offset := int64(8 + headerSize + tensor.DataOffset[0])
	return &SafetensorsTensorReader{SectionReader: io.NewSectionReader(file, offset, int64(tensor.ByteSize)), file: file}, tensor, nil
}

func (s *SafetensorsStore) LoadTensorFloat32(name string) ([]float32, SafetensorsTensor, error) {
	data, tensor, err := s.ReadTensor(name)
	if err != nil {
		return nil, SafetensorsTensor{}, err
	}
	values, err := safetensorsFloat32(tensor, data)
	if err != nil {
		return nil, SafetensorsTensor{}, err
	}
	return values, tensor, nil
}

func (s *SafetensorsStore) findTensor(name string) (SafetensorsTensor, uint64, string, error) {
	tensor, err := s.tensorByName(name)
	if err != nil {
		return SafetensorsTensor{}, 0, "", err
	}
	for _, file := range s.info.Files {
		if file.Path != tensor.File {
			continue
		}
		path := filepath.Join(s.rootDir, filepath.FromSlash(file.Path))
		return tensor, file.HeaderSize, path, nil
	}
	return SafetensorsTensor{}, 0, "", fmt.Errorf("safetensors file %q for tensor %q is missing from metadata", tensor.File, name)
}

func (s *SafetensorsStore) tensorByName(name string) (SafetensorsTensor, error) {
	if s == nil || s.info == nil {
		return SafetensorsTensor{}, fmt.Errorf("safetensors store is nil")
	}
	for _, tensor := range s.info.Tensors {
		if tensor.Name == name {
			return tensor, nil
		}
	}
	return SafetensorsTensor{}, fmt.Errorf("tensor %q not found", name)
}

func safetensorsFloat32(tensor SafetensorsTensor, data []byte) ([]float32, error) {
	if tensor.Parameters > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("tensor %q is too large for this runtime", tensor.Name)
	}
	count := int(tensor.Parameters)
	switch tensor.DType {
	case "F32":
		if len(data) != count*4 {
			return nil, fmt.Errorf("tensor %q has %d bytes, want %d", tensor.Name, len(data), count*4)
		}
		out := make([]float32, count)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return out, nil
	case "F16":
		if len(data) != count*2 {
			return nil, fmt.Errorf("tensor %q has %d bytes, want %d", tensor.Name, len(data), count*2)
		}
		out := make([]float32, count)
		for i := range out {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(data[i*2:]))
		}
		return out, nil
	case "BF16":
		if len(data) != count*2 {
			return nil, fmt.Errorf("tensor %q has %d bytes, want %d", tensor.Name, len(data), count*2)
		}
		out := make([]float32, count)
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(data[i*2:])) << 16)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("tensor %q dtype %s cannot be converted to float32", tensor.Name, tensor.DType)
	}
}

func float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := int((bits >> 10) & 0x1f)
	frac := uint32(bits & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &^= 0x0400
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | (frac << 13))
	}
	exp = exp + (127 - 15)
	return math.Float32frombits(sign | (uint32(exp) << 23) | (frac << 13))
}
