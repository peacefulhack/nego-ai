package modelinfo

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

type EncodeOptions struct {
	AddBOS bool
	AddEOS bool
}

func (v *GGUFVocab) Encode(text string, options EncodeOptions) ([]int, error) {
	if v == nil {
		return nil, fmt.Errorf("gguf vocab is nil")
	}
	index := v.tokenIndex()
	var ids []int
	if options.AddBOS {
		if id, ok := specialTokenID(index, v.BOSTokenID, "<s>", "<bos>", "<|begin_of_text|>"); ok {
			ids = append(ids, id)
		}
	}
	for _, segment := range splitEncodeSegments(text) {
		encoded, err := v.encodeSegment(segment, index)
		if err != nil {
			return nil, err
		}
		ids = append(ids, encoded...)
	}
	if options.AddEOS {
		if id, ok := specialTokenID(index, v.EOSTokenID, "</s>", "<eos>", "<|end_of_text|>"); ok {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func (v *GGUFVocab) tokenIndex() map[string]int {
	index := make(map[string]int, len(v.Tokens))
	for id, token := range v.Tokens {
		if _, exists := index[token]; !exists {
			index[token] = id
		}
	}
	return index
}

func (v *GGUFVocab) encodeSegment(segment string, index map[string]int) ([]int, error) {
	if id, ok := index[segment]; ok {
		return []int{id}, nil
	}
	var ids []int
	for len(segment) > 0 {
		token, id, ok := longestVocabToken(segment, index)
		if ok {
			ids = append(ids, id)
			segment = segment[len(token):]
			continue
		}
		r, size := utf8.DecodeRuneInString(segment)
		if r == utf8.RuneError && size == 0 {
			break
		}
		if encodedBytes, ok := encodeByteFallback(segment[:size], index); ok {
			ids = append(ids, encodedBytes...)
			segment = segment[size:]
			continue
		}
		if unkID, ok := specialTokenID(index, v.UNKTokenID, "<unk>", "<UNK>", "<|unknown|>"); ok {
			ids = append(ids, unkID)
			segment = segment[size:]
			continue
		}
		return nil, fmt.Errorf("cannot encode segment %q with GGUF vocab", segment)
	}
	return ids, nil
}

func longestVocabToken(text string, index map[string]int) (string, int, bool) {
	for n := len(text); n > 0; n-- {
		candidate := text[:n]
		if id, ok := index[candidate]; ok {
			return candidate, id, true
		}
		for _, prefixed := range prefixedEncodeCandidates(candidate) {
			if id, ok := index[prefixed]; ok {
				return candidate, id, true
			}
		}
	}
	return "", 0, false
}

func prefixedEncodeCandidates(text string) []string {
	if strings.HasPrefix(text, " ") {
		trimmed := strings.TrimPrefix(text, " ")
		return []string{"▁" + trimmed, "Ġ" + trimmed}
	}
	return nil
}

func specialTokenID(index map[string]int, configured uint32, candidates ...string) (int, bool) {
	if configured != 0 {
		return int(configured), true
	}
	for _, candidate := range candidates {
		if id, ok := index[candidate]; ok {
			return id, true
		}
	}
	return 0, false
}

func encodeByteFallback(text string, index map[string]int) ([]int, bool) {
	var ids []int
	for i := 0; i < len(text); i++ {
		token := fmt.Sprintf("<0x%02X>", text[i])
		id, ok := index[token]
		if !ok {
			return nil, false
		}
		ids = append(ids, id)
	}
	return ids, true
}

func splitEncodeSegments(text string) []string {
	if text == "" {
		return nil
	}
	var segments []string
	start := -1
	inWord := false
	for i, r := range text {
		if r == ' ' || r == '\n' || r == '\t' {
			if inWord {
				segments = append(segments, text[start:i])
				inWord = false
				start = i
			} else if start == -1 {
				start = i
			}
			continue
		}
		if !inWord {
			if start == -1 {
				start = i
			}
			inWord = true
		}
	}
	if start >= 0 && start < len(text) {
		segments = append(segments, text[start:])
	}
	return segments
}
