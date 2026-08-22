package modelinfo

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	ggufMagic             = "GGUF"
	maxGGUFStringLength   = 16 << 20
	maxGGUFArrayItems     = 1_000_000
	maxGGUFArrayStored    = 128
	maxGGUFMetadataFields = 1_000_000
)

type GGUFInfo struct {
	Path            string         `json:"path"`
	Version         uint32         `json:"version"`
	TensorCount     uint64         `json:"tensor_count"`
	MetadataCount   uint64         `json:"metadata_count"`
	Architecture    string         `json:"architecture,omitempty"`
	Quantization    string         `json:"quantization,omitempty"`
	ContextLength   uint64         `json:"context_length,omitempty"`
	EmbeddingLength uint64         `json:"embedding_length,omitempty"`
	BlockCount      uint64         `json:"block_count,omitempty"`
	VocabSize       uint64         `json:"vocab_size,omitempty"`
	ChatTemplate    string         `json:"chat_template,omitempty"`
	Metadata        map[string]any `json:"metadata,omitempty"`
}

type GGUFArray struct {
	ElemType  uint32 `json:"elem_type"`
	Length    uint64 `json:"length"`
	Values    []any  `json:"values,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
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
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read gguf magic: %w", err)
	}
	if string(magic[:]) != ggufMagic {
		return nil, fmt.Errorf("invalid gguf magic %q", string(magic[:]))
	}
	info := &GGUFInfo{Path: path, Metadata: make(map[string]any)}
	if err := binary.Read(r, binary.LittleEndian, &info.Version); err != nil {
		return nil, fmt.Errorf("read gguf version: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &info.TensorCount); err != nil {
		return nil, fmt.Errorf("read gguf tensor count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &info.MetadataCount); err != nil {
		return nil, fmt.Errorf("read gguf metadata count: %w", err)
	}
	if info.MetadataCount > maxGGUFMetadataFields {
		return nil, fmt.Errorf("gguf metadata count %d exceeds limit %d", info.MetadataCount, maxGGUFMetadataFields)
	}
	for i := uint64(0); i < info.MetadataCount; i++ {
		key, err := readGGUFString(r)
		if err != nil {
			return nil, fmt.Errorf("read gguf metadata key %d: %w", i, err)
		}
		value, err := readGGUFValue(r)
		if err != nil {
			return nil, fmt.Errorf("read gguf metadata %q: %w", key, err)
		}
		info.Metadata[key] = value
	}
	info.populateSummary()
	if len(info.Metadata) == 0 {
		info.Metadata = nil
	}
	return info, nil
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
