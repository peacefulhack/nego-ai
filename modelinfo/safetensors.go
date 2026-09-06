package modelinfo

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

const maxSafetensorsHeaderBytes = 100 << 20

type SafetensorsInfo struct {
	Path        string              `json:"path"`
	Files       []SafetensorsFile   `json:"files,omitempty"`
	Tensors     []SafetensorsTensor `json:"tensors,omitempty"`
	Metadata    map[string]string   `json:"metadata,omitempty"`
	DTypeCounts map[string]int      `json:"dtype_counts,omitempty"`
	ParamCount  uint64              `json:"param_count,omitempty"`
	TotalSize   uint64              `json:"total_size,omitempty"`
}

type SafetensorsFile struct {
	Path       string            `json:"path"`
	HeaderSize uint64            `json:"header_size"`
	Tensors    int               `json:"tensors"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type SafetensorsTensor struct {
	Name       string    `json:"name"`
	File       string    `json:"file,omitempty"`
	DType      string    `json:"dtype"`
	Shape      []uint64  `json:"shape"`
	DataOffset [2]uint64 `json:"data_offsets"`
	ByteSize   uint64    `json:"byte_size"`
	Parameters uint64    `json:"parameters"`
}

func InspectSafetensors(path string) (*SafetensorsInfo, error) {
	if path == "" {
		return nil, fmt.Errorf("safetensors path is required")
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	stat, err := os.Stat(absPath)
	if err != nil {
		return nil, err
	}
	info := &SafetensorsInfo{
		Path:        absPath,
		DTypeCounts: make(map[string]int),
	}
	if !stat.IsDir() {
		if fileKind(absPath) != "safetensors" {
			return nil, fmt.Errorf("%q is not a safetensors file", absPath)
		}
		if err := inspectSafetensorsFile(absPath, filepath.Base(absPath), info); err != nil {
			return nil, err
		}
		return info, nil
	}
	files, err := inspectFiles(absPath)
	if err != nil {
		return nil, err
	}
	for _, file := range files {
		if file.Kind != "safetensors" {
			continue
		}
		fullPath := filepath.Join(absPath, filepath.FromSlash(file.Path))
		if err := inspectSafetensorsFile(fullPath, file.Path, info); err != nil {
			return nil, err
		}
	}
	if len(info.Files) == 0 {
		return nil, fmt.Errorf("no safetensors files found under %q", absPath)
	}
	sort.Slice(info.Tensors, func(i, j int) bool {
		if info.Tensors[i].File != info.Tensors[j].File {
			return info.Tensors[i].File < info.Tensors[j].File
		}
		return info.Tensors[i].Name < info.Tensors[j].Name
	})
	return info, nil
}

func inspectSafetensorsFile(path, displayPath string, info *SafetensorsInfo) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return err
	}
	headerSize, tensors, metadata, err := ReadSafetensorsHeader(file, uint64(stat.Size()))
	if err != nil {
		return fmt.Errorf("read safetensors %q: %w", displayPath, err)
	}
	info.Files = append(info.Files, SafetensorsFile{
		Path:       displayPath,
		HeaderSize: headerSize,
		Tensors:    len(tensors),
		Metadata:   metadata,
	})
	if info.Metadata == nil && len(metadata) > 0 {
		info.Metadata = metadata
	}
	for _, tensor := range tensors {
		tensor.File = displayPath
		info.Tensors = append(info.Tensors, tensor)
		info.DTypeCounts[tensor.DType]++
		if info.ParamCount > ^uint64(0)-tensor.Parameters {
			return fmt.Errorf("safetensors parameter count overflows")
		}
		if info.TotalSize > ^uint64(0)-tensor.ByteSize {
			return fmt.Errorf("safetensors byte size overflows")
		}
		info.ParamCount += tensor.Parameters
		info.TotalSize += tensor.ByteSize
	}
	return nil
}

func ReadSafetensorsHeader(r io.Reader, fileSize uint64) (uint64, []SafetensorsTensor, map[string]string, error) {
	var headerSize uint64
	if err := binary.Read(r, binary.LittleEndian, &headerSize); err != nil {
		return 0, nil, nil, fmt.Errorf("read header size: %w", err)
	}
	if headerSize == 0 {
		return 0, nil, nil, fmt.Errorf("header is empty")
	}
	if headerSize > maxSafetensorsHeaderBytes {
		return 0, nil, nil, fmt.Errorf("header size %d exceeds limit %d", headerSize, maxSafetensorsHeaderBytes)
	}
	if fileSize > 0 && fileSize < 8 {
		return 0, nil, nil, fmt.Errorf("file is too small for a safetensors header")
	}
	if fileSize > 0 && headerSize > fileSize-8 {
		return 0, nil, nil, fmt.Errorf("header size %d exceeds file size %d", headerSize, fileSize)
	}
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(r, header); err != nil {
		return 0, nil, nil, fmt.Errorf("read header: %w", err)
	}
	if len(header) == 0 || header[0] != '{' {
		return 0, nil, nil, fmt.Errorf("header must begin with '{'")
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return 0, nil, nil, fmt.Errorf("parse header json: %w", err)
	}
	metadata := make(map[string]string)
	var tensors []SafetensorsTensor
	dataSize := uint64(0)
	if fileSize > 0 {
		dataSize = fileSize - 8 - headerSize
	}
	for name, value := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(value, &metadata); err != nil {
				return 0, nil, nil, fmt.Errorf("parse __metadata__: %w", err)
			}
			continue
		}
		tensor, err := parseSafetensorsTensor(name, value, dataSize)
		if err != nil {
			return 0, nil, nil, err
		}
		tensors = append(tensors, tensor)
	}
	sort.Slice(tensors, func(i, j int) bool {
		return tensors[i].Name < tensors[j].Name
	})
	if len(metadata) == 0 {
		metadata = nil
	}
	return headerSize, tensors, metadata, nil
}

func parseSafetensorsTensor(name string, value json.RawMessage, dataSize uint64) (SafetensorsTensor, error) {
	var raw struct {
		DType       string   `json:"dtype"`
		Shape       []uint64 `json:"shape"`
		DataOffsets []uint64 `json:"data_offsets"`
	}
	if err := json.Unmarshal(value, &raw); err != nil {
		return SafetensorsTensor{}, fmt.Errorf("parse tensor %q: %w", name, err)
	}
	if raw.DType == "" {
		return SafetensorsTensor{}, fmt.Errorf("tensor %q dtype is required", name)
	}
	if len(raw.DataOffsets) != 2 {
		return SafetensorsTensor{}, fmt.Errorf("tensor %q data_offsets must contain two values", name)
	}
	begin, end := raw.DataOffsets[0], raw.DataOffsets[1]
	if end < begin {
		return SafetensorsTensor{}, fmt.Errorf("tensor %q data_offsets are invalid", name)
	}
	if dataSize > 0 && end > dataSize {
		return SafetensorsTensor{}, fmt.Errorf("tensor %q exceeds data buffer", name)
	}
	parameters, err := shapeElementCount(raw.Shape)
	if err != nil {
		return SafetensorsTensor{}, fmt.Errorf("tensor %q: %w", name, err)
	}
	if size, ok := safetensorsDTypeSize(raw.DType); ok && size > 0 {
		if parameters > ^uint64(0)/uint64(size) {
			return SafetensorsTensor{}, fmt.Errorf("tensor %q byte size overflows", name)
		}
		expected := parameters * uint64(size)
		if expected != end-begin {
			return SafetensorsTensor{}, fmt.Errorf("tensor %q byte size is %d, want %d from dtype and shape", name, end-begin, expected)
		}
	}
	return SafetensorsTensor{
		Name:       name,
		DType:      raw.DType,
		Shape:      append([]uint64(nil), raw.Shape...),
		DataOffset: [2]uint64{begin, end},
		ByteSize:   end - begin,
		Parameters: parameters,
	}, nil
}

func shapeElementCount(shape []uint64) (uint64, error) {
	if len(shape) == 0 {
		return 1, nil
	}
	total := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, nil
		}
		if total > ^uint64(0)/dim {
			return 0, fmt.Errorf("shape element count overflows")
		}
		total *= dim
	}
	return total, nil
}

func safetensorsDTypeSize(dtype string) (int, bool) {
	switch dtype {
	case "BOOL", "U8", "I8", "F8_E4M3", "F8_E5M2":
		return 1, true
	case "I16", "U16", "F16", "BF16":
		return 2, true
	case "I32", "U32", "F32":
		return 4, true
	case "I64", "U64", "F64":
		return 8, true
	default:
		return 0, false
	}
}
