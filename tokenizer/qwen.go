package tokenizer

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Qwen2/Qwen3's published pre-tokenizer pattern. Match metadata exactly so a
// different byte-level tokenizer cannot silently take this specialized path.
// Reference: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/tokenization_qwen2.py
const qwenPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type preTokenizerConfig struct {
	Type          string               `json:"type"`
	PreTokenizers []preTokenizerConfig `json:"pretokenizers"`
	Pattern       struct {
		Regex string `json:"Regex"`
	} `json:"pattern"`
	Behavior       string `json:"behavior"`
	Invert         bool   `json:"invert"`
	AddPrefixSpace bool   `json:"add_prefix_space"`
	UseRegex       *bool  `json:"use_regex"`
}

func qwenEncoderConfig(pre preTokenizerConfig, normalizer json.RawMessage) (bool, bool, error) {
	if pre.Type != "Sequence" || len(pre.PreTokenizers) != 2 || pre.PreTokenizers[0].Type != "Split" || pre.PreTokenizers[1].Type != "ByteLevel" {
		return false, false, nil
	}
	split, byteLevel := pre.PreTokenizers[0], pre.PreTokenizers[1]
	if split.Pattern.Regex != qwenPattern {
		return false, false, fmt.Errorf("unsupported Split/ByteLevel pattern: only the Qwen2/Qwen3 pattern is supported")
	}
	if split.Behavior != "Isolated" || split.Invert || byteLevel.AddPrefixSpace || byteLevel.UseRegex == nil || *byteLevel.UseRegex {
		return false, false, fmt.Errorf("unsupported Qwen pre-tokenizer options")
	}
	if len(normalizer) == 0 || string(normalizer) == "null" {
		return true, false, nil
	}
	var cfg struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(normalizer, &cfg); err != nil {
		return false, false, fmt.Errorf("parse Qwen normalizer: %w", err)
	}
	if cfg.Type != "NFC" {
		return false, false, fmt.Errorf("unsupported Qwen normalizer %q", cfg.Type)
	}
	return true, true, nil
}

// Go's regexp has no lookahead and its \s is ASCII-only. The first four
// alternatives use explicit Unicode whitespace; the trailing-whitespace
// alternatives are evaluated below with their original greedy/backtrack order.
var qwenWord = regexp.MustCompile(`^(?:(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\t\n\v\f\r \x{0085}\p{Z}\p{L}\p{N}]+[\r\n]*)`)

func splitQwen(text string) []string {
	var pieces []string
	for text != "" {
		end := 0
		if match := qwenWord.FindStringIndex(text); match != nil {
			end = match[1]
		} else {
			lastNewline, lastStart := 0, 0
			for offset, char := range text {
				if !unicode.IsSpace(char) {
					break
				}
				lastStart = offset
				end = offset + utf8.RuneLen(char)
				if char == '\r' || char == '\n' {
					lastNewline = end
				}
			}
			if lastNewline > 0 {
				end = lastNewline
			} else if end < len(text) && lastStart > 0 {
				end = lastStart
			}
		}
		// Encode rejects invalid UTF-8; this guard also keeps this helper total.
		if end == 0 {
			_, end = utf8.DecodeRuneInString(text)
		}
		pieces = append(pieces, text[:end])
		text = text[end:]
	}
	return pieces
}

var byteLevelAlphabet = func() [256]string {
	var result [256]string
	for char, value := range byteLevelReverse {
		result[value] = string(char)
	}
	return result
}()

func (t *Tokenizer) encodeQwen(text string) ([]int, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("Qwen tokenizer input must be valid UTF-8")
	}
	var ids []int
	for text != "" {
		index, added := len(text), ""
		// Added is sorted longest-first. Choose the earliest match, retaining
		// the longest match at the same position, before normalization/splitting.
		for _, candidate := range t.Added {
			if candidate == "" {
				continue
			}
			if at := strings.Index(text, candidate); at >= 0 && at < index {
				index, added = at, candidate
			}
		}
		plain := text[:index]
		if t.normalizeNFC {
			normalized := norm.NFC.String(plain)
			// x/text inserts CGJ in excessively long combining sequences for
			// stream safety. Reject that case instead of silently changing HF IDs.
			if strings.Count(normalized, "\u034f") > strings.Count(plain, "\u034f") {
				return nil, fmt.Errorf("Qwen NFC normalization exceeds the supported combining-sequence length")
			}
			plain = normalized
		}
		for _, segment := range splitQwen(plain) {
			pieces := make([]string, len(segment))
			for i := range pieces {
				pieces[i] = byteLevelAlphabet[segment[i]]
			}
			tokens, err := t.mergeBPE(pieces)
			if err != nil {
				return nil, err
			}
			ids = append(ids, tokens...)
		}
		if added == "" {
			break
		}
		ids = append(ids, t.Vocab[added])
		text = text[index+len(added):]
	}
	return ids, nil
}
