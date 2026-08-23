package modelinfo

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	ggufMagic              = "GGUF"
	maxGGUFStringLength    = 16 << 20
	maxGGUFArrayItems      = 1_000_000
	maxGGUFArrayStored     = 128
	maxGGUFMetadataFields  = 1_000_000
	maxGGUFVocabTokens     = 2_000_000
	maxGGUFVocabTokenBytes = 256 << 20
)

type GGUFInfo struct {
	Path            string         `json:"path"`
	Version         uint32         `json:"version"`
	TensorCount     uint64         `json:"tensor_count"`
	MetadataCount   uint64         `json:"metadata_count"`
	Alignment       uint64         `json:"alignment,omitempty"`
	DataOffset      uint64         `json:"data_offset,omitempty"`
	Architecture    string         `json:"architecture,omitempty"`
	Quantization    string         `json:"quantization,omitempty"`
	ContextLength   uint64         `json:"context_length,omitempty"`
	EmbeddingLength uint64         `json:"embedding_length,omitempty"`
	BlockCount      uint64         `json:"block_count,omitempty"`
	VocabSize       uint64         `json:"vocab_size,omitempty"`
	ChatTemplate    string         `json:"chat_template,omitempty"`
	Metadata        map[string]any `json:"metadata,omitempty"`
	Tensors         []GGUFTensor   `json:"tensors,omitempty"`
}

type GGUFArray struct {
	ElemType  uint32 `json:"elem_type"`
	Length    uint64 `json:"length"`
	Values    []any  `json:"values,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

type GGUFTensor struct {
	Name           string   `json:"name"`
	Shape          []uint64 `json:"shape,omitempty"`
	GGMLType       uint32   `json:"ggml_type"`
	Type           string   `json:"type"`
	Offset         uint64   `json:"offset"`
	AbsoluteOffset uint64   `json:"absolute_offset,omitempty"`
	ElementCount   uint64   `json:"element_count,omitempty"`
}

func InspectGGUF(path string) (*GGUFInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return ReadGGUF(file, path)
}

func ReadGGUF(r io.Reader, path string) (*GGUFInfo, error) {
	cr := &countingReader{r: r}
	var magic [4]byte
	if _, err := io.ReadFull(cr, magic[:]); err != nil {
		return nil, fmt.Errorf("read gguf magic: %w", err)
	}
	if string(magic[:]) != ggufMagic {
		return nil, fmt.Errorf("invalid gguf magic %q", string(magic[:]))
	}
	info := &GGUFInfo{Path: path, Metadata: make(map[string]any)}
	if err := binary.Read(cr, binary.LittleEndian, &info.Version); err != nil {
		return nil, fmt.Errorf("read gguf version: %w", err)
	}
	if err := binary.Read(cr, binary.LittleEndian, &info.TensorCount); err != nil {
		return nil, fmt.Errorf("read gguf tensor count: %w", err)
	}
	if err := binary.Read(cr, binary.LittleEndian, &info.MetadataCount); err != nil {
		return nil, fmt.Errorf("read gguf metadata count: %w", err)
	}
	if info.MetadataCount > maxGGUFMetadataFields {
		return nil, fmt.Errorf("gguf metadata count %d exceeds limit %d", info.MetadataCount, maxGGUFMetadataFields)
	}
	for i := uint64(0); i < info.MetadataCount; i++ {
		key, err := readGGUFString(cr)
		if err != nil {
			return nil, fmt.Errorf("read gguf metadata key %d: %w", i, err)
		}
		value, err := readGGUFValue(cr)
		if err != nil {
			return nil, fmt.Errorf("read gguf metadata %q: %w", key, err)
		}
		info.Metadata[key] = value
	}
	info.populateSummary()
	if info.TensorCount > 0 {
		tensors, err := readGGUFTensors(cr, info.TensorCount)
		if err != nil {
			return nil, err
		}
		info.Tensors = tensors
	}
	info.Alignment = metadataUint(info.Metadata, "general.alignment")
	if info.Alignment == 0 {
		info.Alignment = 32
	}
	info.DataOffset = alignOffset(cr.n, info.Alignment)
	for i := range info.Tensors {
		if info.Tensors[i].Offset > ^uint64(0)-info.DataOffset {
			return nil, fmt.Errorf("gguf tensor %q offset overflows data section", info.Tensors[i].Name)
		}
		info.Tensors[i].AbsoluteOffset = info.DataOffset + info.Tensors[i].Offset
	}
	if len(info.Metadata) == 0 {
		info.Metadata = nil
	}
	return info, nil
}

type countingReader struct {
	r io.Reader
	n uint64
}

func (r *countingReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	r.n += uint64(n)
	return n, err
}

func (g *GGUFInfo) populateSummary() {
	g.Architecture = metadataString(g.Metadata, "general.architecture")
	g.Quantization = metadataString(g.Metadata, "general.file_type")
	g.ChatTemplate = metadataString(g.Metadata, "tokenizer.chat_template")
	if g.Architecture != "" {
		prefix := g.Architecture + "."
		g.ContextLength = metadataUint(g.Metadata, prefix+"context_length")
		g.EmbeddingLength = metadataUint(g.Metadata, prefix+"embedding_length")
		g.BlockCount = metadataUint(g.Metadata, prefix+"block_count")
	}
	g.VocabSize = metadataArrayLen(g.Metadata, "tokenizer.ggml.tokens")
}

func readGGUFValue(r io.Reader) (any, error) {
	var typ uint32
	if err := binary.Read(r, binary.LittleEndian, &typ); err != nil {
		return nil, err
	}
	return readGGUFValuePayload(r, typ)
}

func readGGUFValuePayload(r io.Reader, typ uint32) (any, error) {
	switch typ {
	case 0:
		var value uint8
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 1:
		var value int8
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 2:
		var value uint16
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 3:
		var value int16
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 4:
		var value uint32
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 5:
		var value int32
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 6:
		var value float32
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 7:
		var value uint8
		if err := binary.Read(r, binary.LittleEndian, &value); err != nil {
			return nil, err
		}
		return value != 0, nil
	case 8:
		return readGGUFString(r)
	case 9:
		return readGGUFArray(r)
	case 10:
		var value uint64
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 11:
		var value int64
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 12:
		var value float64
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	default:
		return nil, fmt.Errorf("unsupported gguf value type %d", typ)
	}
}

func readGGUFArray(r io.Reader) (GGUFArray, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return GGUFArray{}, err
	}
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return GGUFArray{}, err
	}
	if length > maxGGUFArrayItems {
		return GGUFArray{}, fmt.Errorf("gguf array length %d exceeds limit %d", length, maxGGUFArrayItems)
	}
	out := GGUFArray{
		ElemType:  elemType,
		Length:    length,
		Truncated: length > maxGGUFArrayStored,
	}
	if length <= maxGGUFArrayStored {
		out.Values = make([]any, 0, length)
	}
	for i := uint64(0); i < length; i++ {
		value, err := readGGUFValueOfType(r, elemType)
		if err != nil {
			return GGUFArray{}, err
		}
		if i < maxGGUFArrayStored {
			out.Values = append(out.Values, value)
		}
	}
	return out, nil
}

func readGGUFTensors(r io.Reader, count uint64) ([]GGUFTensor, error) {
	if count > maxGGUFMetadataFields {
		return nil, fmt.Errorf("gguf tensor count %d exceeds limit %d", count, maxGGUFMetadataFields)
	}
	tensors := make([]GGUFTensor, 0, count)
	for i := uint64(0); i < count; i++ {
		tensor, err := readGGUFTensor(r)
		if err != nil {
			return nil, fmt.Errorf("read gguf tensor %d: %w", i, err)
		}
		tensors = append(tensors, tensor)
	}
	return tensors, nil
}

func readGGUFTensor(r io.Reader) (GGUFTensor, error) {
	name, err := readGGUFString(r)
	if err != nil {
		return GGUFTensor{}, err
	}
	var dims uint32
	if err := binary.Read(r, binary.LittleEndian, &dims); err != nil {
		return GGUFTensor{}, err
	}
	if dims > 16 {
		return GGUFTensor{}, fmt.Errorf("tensor %q has too many dimensions: %d", name, dims)
	}
	shape := make([]uint64, dims)
	elementCount := uint64(1)
	for i := range shape {
		if err := binary.Read(r, binary.LittleEndian, &shape[i]); err != nil {
			return GGUFTensor{}, err
		}
		if shape[i] == 0 {
			elementCount = 0
		} else if elementCount != 0 {
			if elementCount > ^uint64(0)/shape[i] {
				return GGUFTensor{}, fmt.Errorf("tensor %q element count overflows", name)
			}
			elementCount *= shape[i]
		}
	}
	var typ uint32
	if err := binary.Read(r, binary.LittleEndian, &typ); err != nil {
		return GGUFTensor{}, err
	}
	var offset uint64
	if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
		return GGUFTensor{}, err
	}
	return GGUFTensor{
		Name:         name,
		Shape:        shape,
		GGMLType:     typ,
		Type:         ggmlTypeName(typ),
		Offset:       offset,
		ElementCount: elementCount,
	}, nil
}

func readGGUFValueOfType(r io.Reader, typ uint32) (any, error) {
	switch typ {
	case 0:
		var value uint8
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 1:
		var value int8
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 2:
		var value uint16
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 3:
		var value int16
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 4:
		var value uint32
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 5:
		var value int32
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 6:
		var value float32
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 7:
		var value uint8
		if err := binary.Read(r, binary.LittleEndian, &value); err != nil {
			return nil, err
		}
		return value != 0, nil
	case 8:
		return readGGUFString(r)
	case 10:
		var value uint64
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 11:
		var value int64
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	case 12:
		var value float64
		err := binary.Read(r, binary.LittleEndian, &value)
		return value, err
	default:
		return nil, fmt.Errorf("unsupported gguf array value type %d", typ)
	}
}

func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > maxGGUFStringLength {
		return "", fmt.Errorf("gguf string length %d exceeds limit %d", length, maxGGUFStringLength)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func metadataString(metadata map[string]any, key string) string {
	value, ok := metadata[key]
	if !ok {
		return ""
	}
	switch value := value.(type) {
	case string:
		return value
	case uint32:
		return ggufFileTypeName(value)
	case uint64:
		return ggufFileTypeName(uint32(value))
	default:
		return ""
	}
}

func metadataUint(metadata map[string]any, key string) uint64 {
	switch value := metadata[key].(type) {
	case uint8:
		return uint64(value)
	case uint16:
		return uint64(value)
	case uint32:
		return uint64(value)
	case uint64:
		return value
	case int8:
		if value >= 0 {
			return uint64(value)
		}
	case int16:
		if value >= 0 {
			return uint64(value)
		}
	case int32:
		if value >= 0 {
			return uint64(value)
		}
	case int64:
		if value >= 0 {
			return uint64(value)
		}
	}
	return 0
}

func metadataArrayLen(metadata map[string]any, key string) uint64 {
	switch value := metadata[key].(type) {
	case GGUFArray:
		return value.Length
	case []any:
		return uint64(len(value))
	default:
		return 0
	}
}

func ggufFileTypeName(value uint32) string {
	names := map[uint32]string{
		0:  "all_f32",
		1:  "mostly_f16",
		2:  "mostly_q4_0",
		3:  "mostly_q4_1",
		7:  "mostly_q8_0",
		8:  "mostly_q5_0",
		9:  "mostly_q5_1",
		10: "mostly_q2_k",
		11: "mostly_q3_k_s",
		12: "mostly_q3_k_m",
		13: "mostly_q3_k_l",
		14: "mostly_q4_k_s",
		15: "mostly_q4_k_m",
		16: "mostly_q5_k_s",
		17: "mostly_q5_k_m",
		18: "mostly_q6_k",
		19: "mostly_iq2_xxs",
		20: "mostly_iq2_xs",
		21: "mostly_q2_k_s",
		22: "mostly_iq3_xxs",
		23: "mostly_iq1_s",
		24: "mostly_iq4_nl",
		25: "mostly_iq3_s",
		26: "mostly_iq3_m",
		27: "mostly_iq2_s",
		28: "mostly_iq2_m",
		29: "mostly_iq4_xs",
		30: "mostly_iq1_m",
		31: "mostly_bf16",
		32: "mostly_q4_0_4_4",
		33: "mostly_q4_0_4_8",
		34: "mostly_q4_0_8_8",
	}
	if name, ok := names[value]; ok {
		return name
	}
	return strings.TrimSpace(fmt.Sprintf("file_type_%d", value))
}

func ggmlTypeName(value uint32) string {
	names := map[uint32]string{
		0:  "f32",
		1:  "f16",
		2:  "q4_0",
		3:  "q4_1",
		6:  "q5_0",
		7:  "q5_1",
		8:  "q8_0",
		9:  "q8_1",
		10: "q2_k",
		11: "q3_k",
		12: "q4_k",
		13: "q5_k",
		14: "q6_k",
		15: "q8_k",
		16: "iq2_xxs",
		17: "iq2_xs",
		18: "iq3_xxs",
		19: "iq1_s",
		20: "iq4_nl",
		21: "iq3_s",
		22: "iq2_s",
		23: "iq4_xs",
		24: "i8",
		25: "i16",
		26: "i32",
		27: "i64",
		28: "f64",
		29: "iq1_m",
		30: "bf16",
	}
	if name, ok := names[value]; ok {
		return name
	}
	return fmt.Sprintf("ggml_type_%d", value)
}

func alignOffset(offset, alignment uint64) uint64 {
	if alignment == 0 {
		return offset
	}
	remainder := offset % alignment
	if remainder == 0 {
		return offset
	}
	return offset + alignment - remainder
}
