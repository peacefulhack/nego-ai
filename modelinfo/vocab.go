package modelinfo

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type GGUFVocab struct {
	Path         string    `json:"path"`
	Model        string    `json:"model,omitempty"`
	Tokens       []string  `json:"tokens,omitempty"`
	Scores       []float32 `json:"scores,omitempty"`
	TokenTypes   []uint32  `json:"token_types,omitempty"`
	Merges       []string  `json:"merges,omitempty"`
	BOSTokenID   uint32    `json:"bos_token_id,omitempty"`
	EOSTokenID   uint32    `json:"eos_token_id,omitempty"`
	UNKTokenID   uint32    `json:"unk_token_id,omitempty"`
	PADTokenID   uint32    `json:"pad_token_id,omitempty"`
	ChatTemplate string    `json:"chat_template,omitempty"`
}

func InspectGGUFVocab(path string) (*GGUFVocab, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return ReadGGUFVocab(file, path)
}

func ReadGGUFVocab(r io.Reader, path string) (*GGUFVocab, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read gguf magic: %w", err)
	}
	if string(magic[:]) != ggufMagic {
		return nil, fmt.Errorf("invalid gguf magic %q", string(magic[:]))
	}
	var version uint32
	var tensorCount, metadataCount uint64
	for _, target := range []any{&version, &tensorCount, &metadataCount} {
		if err := binary.Read(r, binary.LittleEndian, target); err != nil {
			return nil, fmt.Errorf("read gguf header: %w", err)
		}
	}
	if metadataCount > maxGGUFMetadataFields {
		return nil, fmt.Errorf("gguf metadata count %d exceeds limit %d", metadataCount, maxGGUFMetadataFields)
	}
	vocab := &GGUFVocab{Path: path}
	for i := uint64(0); i < metadataCount; i++ {
		key, err := readGGUFString(r)
		if err != nil {
			return nil, fmt.Errorf("read gguf metadata key %d: %w", i, err)
		}
		var typ uint32
		if err := binary.Read(r, binary.LittleEndian, &typ); err != nil {
			return nil, fmt.Errorf("read gguf metadata type %q: %w", key, err)
		}
		if err := readGGUFVocabValue(r, key, typ, vocab); err != nil {
			return nil, err
		}
	}
	return vocab, nil
}

func readGGUFVocabValue(r io.Reader, key string, typ uint32, vocab *GGUFVocab) error {
	switch key {
	case "tokenizer.ggml.model":
		value, err := readGGUFStringValue(r, typ, key)
		if err != nil {
			return err
		}
		vocab.Model = value
	case "tokenizer.ggml.tokens":
		tokens, err := readGGUFStringArrayValue(r, typ, key)
		if err != nil {
			return err
		}
		vocab.Tokens = tokens
	case "tokenizer.ggml.scores":
		scores, err := readGGUFFloat32ArrayValue(r, typ, key)
		if err != nil {
			return err
		}
		vocab.Scores = scores
	case "tokenizer.ggml.token_type":
		types, err := readGGUFUint32ArrayValue(r, typ, key)
		if err != nil {
			return err
		}
		vocab.TokenTypes = types
	case "tokenizer.ggml.merges":
		merges, err := readGGUFStringArrayValue(r, typ, key)
		if err != nil {
			return err
		}
		vocab.Merges = merges
	case "tokenizer.ggml.bos_token_id":
		value, err := readGGUFUint32Value(r, typ, key)
		if err != nil {
			return err
		}
		vocab.BOSTokenID = value
	case "tokenizer.ggml.eos_token_id":
		value, err := readGGUFUint32Value(r, typ, key)
		if err != nil {
			return err
		}
		vocab.EOSTokenID = value
	case "tokenizer.ggml.unknown_token_id":
		value, err := readGGUFUint32Value(r, typ, key)
		if err != nil {
			return err
		}
		vocab.UNKTokenID = value
	case "tokenizer.ggml.padding_token_id":
		value, err := readGGUFUint32Value(r, typ, key)
		if err != nil {
			return err
		}
		vocab.PADTokenID = value
	case "tokenizer.chat_template":
		value, err := readGGUFStringValue(r, typ, key)
		if err != nil {
			return err
		}
		vocab.ChatTemplate = value
	default:
		if _, err := readGGUFValuePayload(r, typ); err != nil {
			return fmt.Errorf("read gguf metadata %q: %w", key, err)
		}
	}
	return nil
}

func readGGUFStringValue(r io.Reader, typ uint32, key string) (string, error) {
	if typ != 8 {
		return "", fmt.Errorf("gguf metadata %q is type %d, want string", key, typ)
	}
	return readGGUFString(r)
}

func readGGUFUint32Value(r io.Reader, typ uint32, key string) (uint32, error) {
	value, err := readGGUFValuePayload(r, typ)
	if err != nil {
		return 0, fmt.Errorf("read gguf metadata %q: %w", key, err)
	}
	switch value := value.(type) {
	case uint8:
		return uint32(value), nil
	case uint16:
		return uint32(value), nil
	case uint32:
		return value, nil
	case uint64:
		if value > uint64(^uint32(0)) {
			return 0, fmt.Errorf("gguf metadata %q overflows uint32", key)
		}
		return uint32(value), nil
	default:
		return 0, fmt.Errorf("gguf metadata %q is type %T, want unsigned integer", key, value)
	}
}

func readGGUFStringArrayValue(r io.Reader, typ uint32, key string) ([]string, error) {
	if typ != 9 {
		return nil, fmt.Errorf("gguf metadata %q is type %d, want array", key, typ)
	}
	elemType, length, err := readGGUFArrayHeader(r, key)
	if err != nil {
		return nil, err
	}
	if elemType != 8 {
		return nil, fmt.Errorf("gguf metadata %q array elem type is %d, want string", key, elemType)
	}
	out := make([]string, 0, length)
	var totalBytes uint64
	for i := uint64(0); i < length; i++ {
		value, err := readGGUFString(r)
		if err != nil {
			return nil, fmt.Errorf("read gguf metadata %q item %d: %w", key, i, err)
		}
		if uint64(len(value)) > maxGGUFVocabTokenBytes-totalBytes {
			return nil, fmt.Errorf("gguf metadata %q total token bytes exceed limit %d", key, maxGGUFVocabTokenBytes)
		}
		totalBytes += uint64(len(value))
		out = append(out, value)
	}
	return out, nil
}

func readGGUFFloat32ArrayValue(r io.Reader, typ uint32, key string) ([]float32, error) {
	if typ != 9 {
		return nil, fmt.Errorf("gguf metadata %q is type %d, want array", key, typ)
	}
	elemType, length, err := readGGUFArrayHeader(r, key)
	if err != nil {
		return nil, err
	}
	if elemType != 6 {
		return nil, fmt.Errorf("gguf metadata %q array elem type is %d, want float32", key, elemType)
	}
	out := make([]float32, 0, length)
	for i := uint64(0); i < length; i++ {
		var value float32
		if err := binary.Read(r, binary.LittleEndian, &value); err != nil {
			return nil, fmt.Errorf("read gguf metadata %q item %d: %w", key, i, err)
		}
		out = append(out, value)
	}
	return out, nil
}

func readGGUFUint32ArrayValue(r io.Reader, typ uint32, key string) ([]uint32, error) {
	if typ != 9 {
		return nil, fmt.Errorf("gguf metadata %q is type %d, want array", key, typ)
	}
	elemType, length, err := readGGUFArrayHeader(r, key)
	if err != nil {
		return nil, err
	}
	if elemType != 4 {
		return nil, fmt.Errorf("gguf metadata %q array elem type is %d, want uint32", key, elemType)
	}
	out := make([]uint32, 0, length)
	for i := uint64(0); i < length; i++ {
		var value uint32
		if err := binary.Read(r, binary.LittleEndian, &value); err != nil {
			return nil, fmt.Errorf("read gguf metadata %q item %d: %w", key, i, err)
		}
		out = append(out, value)
	}
	return out, nil
}

func readGGUFArrayHeader(r io.Reader, key string) (uint32, uint64, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return 0, 0, fmt.Errorf("read gguf metadata %q array type: %w", key, err)
	}
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return 0, 0, fmt.Errorf("read gguf metadata %q array length: %w", key, err)
	}
	if length > maxGGUFVocabTokens {
		return 0, 0, fmt.Errorf("gguf metadata %q array length %d exceeds limit %d", key, length, maxGGUFVocabTokens)
	}
	return elemType, length, nil
}
