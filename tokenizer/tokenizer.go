package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

type Tokenizer struct {
	Path       string
	ModelType  string
	UnkToken   string
	Vocab      map[string]int
	IDToToken  map[int]string
	maxTokenLn int
}

func Load(path string) (*Tokenizer, error) {
	tokenizerPath := path
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		tokenizerPath = filepath.Join(path, "tokenizer.json")
	}
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Model struct {
			Type     string         `json:"type"`
			Vocab    map[string]int `json:"vocab"`
			UnkToken string         `json:"unk_token"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if len(raw.Model.Vocab) == 0 && len(raw.AddedTokens) == 0 {
		return nil, fmt.Errorf("tokenizer %s does not contain a vocab", tokenizerPath)
	}
	vocab := make(map[string]int, len(raw.Model.Vocab)+len(raw.AddedTokens))
	for token, id := range raw.Model.Vocab {
		vocab[token] = id
	}
	for _, token := range raw.AddedTokens {
		if token.Content != "" {
			vocab[token.Content] = token.ID
		}
	}
	out := &Tokenizer{
		Path:      tokenizerPath,
		ModelType: raw.Model.Type,
		UnkToken:  raw.Model.UnkToken,
		Vocab:     vocab,
		IDToToken: make(map[int]string, len(vocab)),
	}
	for token, id := range vocab {
		out.IDToToken[id] = token
		if len(token) > out.maxTokenLn {
			out.maxTokenLn = len(token)
		}
	}
	return out, nil
}

func (t *Tokenizer) Encode(text string) ([]int, error) {
	var ids []int
	for _, segment := range splitSegments(text) {
		tokenIDs, err := t.encodeSegment(segment)
		if err != nil {
			return nil, err
		}
		ids = append(ids, tokenIDs...)
	}
	return ids, nil
}

func (t *Tokenizer) Decode(ids []int) (string, error) {
	var b strings.Builder
	for _, id := range ids {
		token, ok := t.IDToToken[id]
		if !ok {
			return "", fmt.Errorf("unknown token id %d", id)
		}
		b.WriteString(decodeToken(token))
	}
	return b.String(), nil
}

func (t *Tokenizer) Count(text string) (int, error) {
	ids, err := t.Encode(text)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

func (t *Tokenizer) encodeSegment(segment string) ([]int, error) {
	if id, ok := t.Vocab[segment]; ok {
		return []int{id}, nil
	}
	var ids []int
	for len(segment) > 0 {
		token, id, ok := t.longestToken(segment)
		if !ok {
			unkID, hasUnk := t.Vocab[t.UnkToken]
			if !hasUnk {
				return nil, fmt.Errorf("cannot encode segment %q", segment)
			}
			_, size := utf8.DecodeRuneInString(segment)
			if size <= 0 {
				size = 1
			}
			ids = append(ids, unkID)
			segment = segment[size:]
			continue
		}
		ids = append(ids, id)
		segment = segment[len(token):]
	}
	return ids, nil
}

func (t *Tokenizer) longestToken(text string) (string, int, bool) {
	maxLen := min(len(text), t.maxTokenLn)
	for n := maxLen; n > 0; n-- {
		candidate := text[:n]
		if id, ok := t.Vocab[candidate]; ok {
			return candidate, id, true
		}
		for _, prefixed := range prefixedCandidates(candidate) {
			if id, ok := t.Vocab[prefixed]; ok {
				return candidate, id, true
			}
		}
	}
	return "", 0, false
}

func prefixedCandidates(text string) []string {
	if strings.HasPrefix(text, " ") {
		trimmed := strings.TrimPrefix(text, " ")
		return []string{"Ġ" + trimmed, "▁" + trimmed}
	}
	return []string{"##" + text}
}

func splitSegments(text string) []string {
	if text == "" {
		return nil
	}
	var segments []string
	start := -1
	inWord := false
	for i, r := range text {
		if isSpace(r) {
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

func isSpace(r rune) bool {
	return r == ' ' || r == '\n' || r == '\t'
}

func decodeToken(token string) string {
	switch {
	case strings.HasPrefix(token, "Ġ"):
		return " " + strings.TrimPrefix(token, "Ġ")
	case strings.HasPrefix(token, "▁"):
		return " " + strings.TrimPrefix(token, "▁")
	case strings.HasPrefix(token, "##"):
		return strings.TrimPrefix(token, "##")
	default:
		return token
	}
}

func (t *Tokenizer) SortedVocabTokens() []string {
	tokens := make([]string, 0, len(t.Vocab))
	for token := range t.Vocab {
		tokens = append(tokens, token)
	}
	sort.Strings(tokens)
	return tokens
}
