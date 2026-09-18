package tokenizer

import (
	"fmt"
	"sort"
	"strings"
)

// QwenOptions describes a Qwen2/Qwen3 byte-level BPE vocabulary without requiring
// tokenizer.json. Token IDs are their positions in Tokens. Inputs are copied.
type QwenOptions struct {
	Tokens        []string
	Merges        []string
	AddedTokenIDs []int
	NormalizeNFC  bool
}

// NewQwen constructs the supported Qwen text splitter, BPE encoder, and byte-level
// decoder. Added tokens match literally before splitting. NFC is opt-in because
// GGUF Qwen tokenization does not declare the HF tokenizer's NFC normalizer.
func NewQwen(options QwenOptions) (*Tokenizer, error) {
	if len(options.Tokens) == 0 {
		return nil, fmt.Errorf("Qwen vocabulary is empty")
	}
	t := &Tokenizer{
		ModelType: "BPE", Vocab: make(map[string]int, len(options.Tokens)),
		IDToToken: make(map[int]string, len(options.Tokens)), Merges: make(map[string]int, len(options.Merges)),
		qwenEncoder: true, byteLevelDecoder: true, normalizeNFC: options.NormalizeNFC,
	}
	for id, token := range options.Tokens {
		if token == "" {
			return nil, fmt.Errorf("Qwen token %d is empty", id)
		}
		if _, exists := t.Vocab[token]; exists {
			return nil, fmt.Errorf("duplicate Qwen token at id %d", id)
		}
		t.Vocab[token], t.IDToToken[id] = id, token
		if len(token) > t.maxTokenLn {
			t.maxTokenLn = len(token)
		}
	}
	for rank, merge := range options.Merges {
		parts := strings.Split(merge, " ")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("invalid Qwen merge at rank %d", rank)
		}
		for _, part := range []string{parts[0], parts[1], parts[0] + parts[1]} {
			if _, ok := t.Vocab[part]; !ok {
				return nil, fmt.Errorf("Qwen merge at rank %d references a missing token", rank)
			}
		}
		if _, exists := t.Merges[merge]; !exists {
			t.Merges[merge] = rank
		}
	}
	seen := make(map[int]bool)
	for _, id := range options.AddedTokenIDs {
		token, ok := t.IDToToken[id]
		if !ok {
			return nil, fmt.Errorf("Qwen added token id %d is out of range", id)
		}
		if !seen[id] {
			t.Added = append(t.Added, token)
			seen[id] = true
		}
	}
	sort.SliceStable(t.Added, func(i, j int) bool { return len(t.Added[i]) > len(t.Added[j]) })
	return t, nil
}
