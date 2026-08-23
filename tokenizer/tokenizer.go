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
	Added      []string
	Merges     map[string]int
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
			Type     string          `json:"type"`
			Vocab    json.RawMessage `json:"vocab"`
			Merges   json.RawMessage `json:"merges"`
			UnkToken string          `json:"unk_token"`
			UnkID    *int            `json:"unk_id"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	modelVocab, err := parseVocab(raw.Model.Vocab)
	if err != nil {
		return nil, err
	}
	if len(modelVocab) == 0 && len(raw.AddedTokens) == 0 {
		return nil, fmt.Errorf("tokenizer %s does not contain a vocab", tokenizerPath)
	}
	merges, err := parseMerges(raw.Model.Merges)
	if err != nil {
		return nil, err
	}
	vocab := make(map[string]int, len(modelVocab)+len(raw.AddedTokens))
	for token, id := range modelVocab {
		vocab[token] = id
	}
	var added []string
	for _, token := range raw.AddedTokens {
		if token.Content != "" {
			vocab[token.Content] = token.ID
			added = append(added, token.Content)
		}
	}
	unkToken := raw.Model.UnkToken
	if unkToken == "" && raw.Model.UnkID != nil {
		for token, id := range vocab {
			if id == *raw.Model.UnkID {
				unkToken = token
				break
			}
		}
	}
	sort.SliceStable(added, func(i, j int) bool {
		return len(added[i]) > len(added[j])
	})
	out := &Tokenizer{
		Path:      tokenizerPath,
		ModelType: raw.Model.Type,
		UnkToken:  unkToken,
		Vocab:     vocab,
		IDToToken: make(map[int]string, len(vocab)),
		Added:     added,
		Merges:    merges,
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

func (t *Tokenizer) EncodeBatch(texts []string) ([][]int, error) {
	out := make([][]int, 0, len(texts))
	for _, text := range texts {
		ids, err := t.Encode(text)
		if err != nil {
			return nil, err
		}
		out = append(out, ids)
	}
	return out, nil
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

func (t *Tokenizer) DecodeBatch(batch [][]int) ([]string, error) {
	out := make([]string, 0, len(batch))
	for _, ids := range batch {
		text, err := t.Decode(ids)
		if err != nil {
			return nil, err
		}
		out = append(out, text)
	}
	return out, nil
}

func (t *Tokenizer) Count(text string) (int, error) {
	ids, err := t.Encode(text)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

func (t *Tokenizer) CountBatch(texts []string) ([]int, error) {
	out := make([]int, 0, len(texts))
	for _, text := range texts {
		count, err := t.Count(text)
		if err != nil {
			return nil, err
		}
		out = append(out, count)
	}
	return out, nil
}

func (t *Tokenizer) encodeSegment(segment string) ([]int, error) {
	if ids, ok, err := t.encodeAddedPieces(segment); ok || err != nil {
		return ids, err
	}
	if strings.EqualFold(t.ModelType, "BPE") && len(t.Merges) > 0 {
		return t.encodeBPE(segment)
	}
	if id, ok := t.Vocab[segment]; ok {
		return []int{id}, nil
	}
	return t.encodeGreedy(segment)
}

func (t *Tokenizer) encodeAddedPieces(segment string) ([]int, bool, error) {
	for _, added := range t.Added {
		if added == "" || !strings.Contains(segment, added) {
			continue
		}
		var ids []int
		for segment != "" {
			idx := strings.Index(segment, added)
			if idx < 0 {
				partIDs, err := t.encodeWithoutAdded(segment)
				if err != nil {
					return nil, true, err
				}
				ids = append(ids, partIDs...)
				break
			}
			if idx > 0 {
				partIDs, err := t.encodeWithoutAdded(segment[:idx])
				if err != nil {
					return nil, true, err
				}
				ids = append(ids, partIDs...)
			}
			ids = append(ids, t.Vocab[added])
			segment = segment[idx+len(added):]
		}
		return ids, true, nil
	}
	return nil, false, nil
}

func (t *Tokenizer) encodeWithoutAdded(segment string) ([]int, error) {
	if strings.EqualFold(t.ModelType, "BPE") && len(t.Merges) > 0 {
		return t.encodeBPE(segment)
	}
	if id, ok := t.Vocab[segment]; ok {
		return []int{id}, nil
	}
	return t.encodeGreedy(segment)
}

func (t *Tokenizer) encodeGreedy(segment string) ([]int, error) {
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

func (t *Tokenizer) encodeBPE(segment string) ([]int, error) {
	pieces := t.bpeInitialPieces(segment)
	for {
		bestIndex := -1
		bestRank := int(^uint(0) >> 1)
		for i := 0; i+1 < len(pieces); i++ {
			joined := pieces[i] + pieces[i+1]
			if _, ok := t.Vocab[joined]; !ok {
				continue
			}
			if rank, ok := t.Merges[pieces[i]+" "+pieces[i+1]]; ok && rank < bestRank {
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
		if id, ok := t.Vocab[piece]; ok {
			ids = append(ids, id)
			continue
		}
		unkID, hasUnk := t.Vocab[t.UnkToken]
		if !hasUnk {
			return nil, fmt.Errorf("cannot encode BPE piece %q", piece)
		}
		ids = append(ids, unkID)
	}
	return ids, nil
}

func (t *Tokenizer) bpeInitialPieces(segment string) []string {
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
		case t.hasToken("Ġ" + first):
			pieces = append(pieces, "Ġ"+first)
		case t.hasToken("▁" + first):
			pieces = append(pieces, "▁"+first)
		case t.hasToken("Ġ"):
			pieces = append(pieces, "Ġ", first)
		case t.hasToken("▁"):
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

func (t *Tokenizer) hasToken(token string) bool {
	_, ok := t.Vocab[token]
	return ok
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
	return []string{"▁" + text, "Ġ" + text, "##" + text}
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

func parseVocab(data json.RawMessage) (map[string]int, error) {
	if len(data) == 0 || string(data) == "null" {
		return nil, nil
	}
	var mapped map[string]int
	if err := json.Unmarshal(data, &mapped); err == nil {
		return mapped, nil
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("parse tokenizer vocab: %w", err)
	}
	vocab := make(map[string]int, len(entries))
	for id, entry := range entries {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) == 0 {
			return nil, fmt.Errorf("parse tokenizer vocab entry %d", id)
		}
		var token string
		if err := json.Unmarshal(pair[0], &token); err != nil {
			return nil, fmt.Errorf("parse tokenizer vocab token %d: %w", id, err)
		}
		vocab[token] = id
	}
	return vocab, nil
}

func parseMerges(data json.RawMessage) (map[string]int, error) {
	if len(data) == 0 || string(data) == "null" {
		return nil, nil
	}
	var rows []json.RawMessage
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, fmt.Errorf("parse tokenizer merges: %w", err)
	}
	merges := make(map[string]int, len(rows))
	for rank, row := range rows {
		var text string
		if err := json.Unmarshal(row, &text); err == nil {
			fields := strings.Fields(text)
			if len(fields) == 2 {
				merges[fields[0]+" "+fields[1]] = rank
				continue
			}
		}
		var pair []string
		if err := json.Unmarshal(row, &pair); err == nil && len(pair) == 2 {
			merges[pair[0]+" "+pair[1]] = rank
			continue
		}
		return nil, fmt.Errorf("parse tokenizer merge %d", rank)
	}
	return merges, nil
}

func (t *Tokenizer) SortedVocabTokens() []string {
	tokens := make([]string, 0, len(t.Vocab))
	for token := range t.Vocab {
		tokens = append(tokens, token)
	}
	sort.Strings(tokens)
	return tokens
}
