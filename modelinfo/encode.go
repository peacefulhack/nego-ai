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
	merges := v.mergeRanks()
	var ids []int
	if options.AddBOS {
		if id, ok := specialTokenID(index, v.BOSTokenID, "<s>", "<bos>", "<|begin_of_text|>"); ok {
			ids = append(ids, id)
		}
	}
	for _, segment := range splitEncodeSegments(text) {
		if len(merges) > 0 {
			encoded, ok, err := v.encodeBPESegment(segment, index, merges)
			if err != nil {
				return nil, err
			}
			if ok {
				ids = append(ids, encoded...)
				continue
			}
		}
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

func (v *GGUFVocab) mergeRanks() map[string]int {
	if len(v.Merges) == 0 {
		return nil
	}
	merges := make(map[string]int, len(v.Merges))
	for rank, merge := range v.Merges {
		fields := strings.Fields(merge)
		if len(fields) == 2 {
			merges[fields[0]+" "+fields[1]] = rank
		}
	}
	return merges
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

func (v *GGUFVocab) encodeBPESegment(segment string, index map[string]int, merges map[string]int) ([]int, bool, error) {
	pieces := bpeInitialPieces(segment, index)
	if len(pieces) == 0 {
		return nil, false, nil
	}
	for {
		bestIndex := -1
		bestRank := int(^uint(0) >> 1)
		for i := 0; i+1 < len(pieces); i++ {
			joined := pieces[i] + pieces[i+1]
			if _, ok := index[joined]; !ok {
				continue
			}
			if rank, ok := merges[pieces[i]+" "+pieces[i+1]]; ok && rank < bestRank {
				bestRank = rank
				bestIndex = i
			}
		}
		if bestIndex < 0 {
			break
		}
		pieces[bestIndex] += pieces[bestIndex+1]
		pieces = append(pieces[:bestIndex+1], pieces[bestIndex+2:]...)
	}
	ids := make([]int, 0, len(pieces))
	for _, piece := range pieces {
		if id, ok := index[piece]; ok {
			ids = append(ids, id)
			continue
		}
		if encodedBytes, ok := encodeByteFallback(piece, index); ok {
			ids = append(ids, encodedBytes...)
			continue
		}
		return nil, false, nil
	}
	return ids, true, nil
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

func bpeInitialPieces(segment string, index map[string]int) []string {
	if segment == "" {
		return nil
	}
	var pieces []string
	if strings.HasPrefix(segment, " ") {
		spaces := len(segment) - len(strings.TrimLeft(segment, " "))
		for i := 0; i < spaces-1; i++ {
			pieces = append(pieces, " ")
		}
		rest := segment[spaces:]
		if rest == "" {
			return append(pieces, " ")
		}
		r, size := utf8.DecodeRuneInString(rest)
		first := string(r)
		switch {
		case hasVocabToken(index, "Ġ"+first):
			pieces = append(pieces, "Ġ"+first)
		case hasVocabToken(index, "▁"+first):
			pieces = append(pieces, "▁"+first)
		case hasVocabToken(index, "Ġ"):
			pieces = append(pieces, "Ġ", first)
		case hasVocabToken(index, "▁"):
			pieces = append(pieces, "▁", first)
		default:
			pieces = append(pieces, " ", first)
		}
		segment = rest[size:]
	}
	for _, r := range segment {
		pieces = append(pieces, string(r))
	}
	return pieces
}

func hasVocabToken(index map[string]int, token string) bool {
	_, ok := index[token]
	return ok
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
