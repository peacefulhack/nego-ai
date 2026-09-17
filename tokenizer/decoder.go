package tokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Decoder assembles token text incrementally. It buffers incomplete UTF-8 byte
// sequences until later tokens complete them. Use one decoder per stream.
type Decoder struct {
	tokenizer *Tokenizer
	pending   []byte
}

// NewDecoder starts an independent decoder for generated token IDs.
func (t *Tokenizer) NewDecoder() *Decoder {
	return &Decoder{tokenizer: t}
}

// Push returns the complete text available after id. It may return an empty
// string when a byte-level token contains only part of a Unicode character.
func (d *Decoder) Push(id int) (string, error) {
	if d == nil || d.tokenizer == nil {
		return "", fmt.Errorf("tokenizer decoder is not initialized")
	}
	token, ok := d.tokenizer.IDToToken[id]
	if !ok {
		return "", fmt.Errorf("unknown token id %d", id)
	}
	if !d.tokenizer.byteLevelDecoder {
		return decodeToken(token), nil
	}
	// ByteLevel decoding reverses the alphabet for every character in a token,
	// including whitespace inside merged tokens. Non-alphabet tokens stay literal.
	start := len(d.pending)
	for _, char := range token {
		value, ok := byteLevelByte(char)
		if !ok {
			d.pending = append(d.pending[:start], token...)
			return d.completeText(false), nil
		}
		d.pending = append(d.pending, value)
	}
	return d.completeText(false), nil
}

// Flush finishes a stream, replacing any incomplete trailing bytes with the
// Unicode replacement character. It drains pending data and may be called again.
func (d *Decoder) Flush() string {
	if d == nil {
		return ""
	}
	return d.completeText(true)
}

func (d *Decoder) completeText(final bool) string {
	var out strings.Builder
	consumed := 0
	for consumed < len(d.pending) {
		remaining := d.pending[consumed:]
		if !utf8.FullRune(remaining) {
			if final {
				out.WriteRune(utf8.RuneError)
				consumed = len(d.pending)
			}
			break
		}
		char, size := utf8.DecodeRune(remaining)
		out.WriteRune(char)
		consumed += size
	}
	copy(d.pending, d.pending[consumed:])
	d.pending = d.pending[:len(d.pending)-consumed]
	return out.String()
}

// The GPT-2/HF ByteLevel alphabet preserves printable Latin-1 bytes, mapping
// the remaining bytes to consecutive code points starting at U+0100.
// Reference: https://github.com/huggingface/tokenizers/blob/v0.21.1/tokenizers/src/pre_tokenizers/byte_level.rs
var byteLevelReverse = func() map[rune]byte {
	result := make(map[rune]byte, 256)
	next := rune(256)
	for value := 0; value < 256; value++ {
		char := rune(value)
		if !(value >= 33 && value <= 126 || value >= 161 && value <= 172 || value >= 174) {
			char = next
			next++
		}
		result[char] = byte(value)
	}
	return result
}()

func byteLevelByte(char rune) (byte, bool) {
	value, ok := byteLevelReverse[char]
	return value, ok
}
