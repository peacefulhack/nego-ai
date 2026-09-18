package modelinfo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/gakon/nego-ai/tokenizer"
)

type DecodeOptions struct {
	SkipSpecial bool
}

func (v *GGUFVocab) Decode(ids []int, options DecodeOptions) (string, error) {
	decoder, err := v.NewDecoder(options)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, id := range ids {
		text, err := decoder.Push(id)
		if err != nil {
			return "", err
		}
		b.WriteString(text)
	}
	b.WriteString(decoder.Flush())
	return b.String(), nil
}

// GGUFDecoder decodes generated tokens. Qwen byte-level decoding buffers partial
// UTF-8 characters across tokens; other vocabularies retain their legacy decoding.
// Create one decoder per stream and call Flush when generation ends.
type GGUFDecoder struct {
	vocab   *GGUFVocab
	options DecodeOptions
	qwen    *tokenizer.Decoder
}

func (v *GGUFVocab) NewDecoder(options DecodeOptions) (*GGUFDecoder, error) {
	if v == nil {
		return nil, fmt.Errorf("gguf vocab is nil")
	}
	d := &GGUFDecoder{vocab: v, options: options}
	if v.PreTokenizer == "qwen2" {
		tok, err := v.qwenTokenizer()
		if err != nil {
			return nil, err
		}
		d.qwen = tok.NewDecoder()
	}
	return d, nil
}

func (d *GGUFDecoder) Push(id int) (string, error) {
	if d == nil || d.vocab == nil {
		return "", fmt.Errorf("GGUF decoder is not initialized")
	}
	if id < 0 || id >= len(d.vocab.Tokens) {
		return "", fmt.Errorf("token id %d is out of range", id)
	}
	if d.options.SkipSpecial && d.vocab.isSpecialToken(id) {
		return "", nil
	}
	if d.qwen != nil {
		return d.qwen.Push(id)
	}
	var b strings.Builder
	writeDecodedToken(&b, d.vocab.Tokens[id])
	return b.String(), nil
}

func (d *GGUFDecoder) Flush() string {
	if d == nil || d.qwen == nil {
		return ""
	}
	return d.qwen.Flush()
}

func (v *GGUFVocab) isSpecialToken(id int) bool {
	if id == int(v.BOSTokenID) && v.BOSTokenID != 0 {
		return true
	}
	if id == int(v.EOSTokenID) && v.EOSTokenID != 0 {
		return true
	}
	if id == int(v.PADTokenID) && v.PADTokenID != 0 {
		return true
	}
	if id == int(v.EOTTokenID) && v.EOTTokenID != 0 {
		return true
	}
	if id == int(v.EOMTokenID) && v.EOMTokenID != 0 {
		return true
	}
	if id < len(v.TokenTypes) && v.TokenTypes[id] == 3 {
		return true
	}
	if id >= 0 && id < len(v.Tokens) && knownSpecialToken(v.Tokens[id]) {
		return true
	}
	return false
}

func knownSpecialToken(token string) bool {
	switch token {
	case "<s>",
		"</s>",
		"<bos>",
		"<eos>",
		"<pad>",
		"<|begin_of_text|>",
		"<|end_of_text|>",
		"<|endoftext|>",
		"<|eot_id|>",
		"<|im_start|>",
		"<|im_end|>":
		return true
	default:
		return false
	}
}

func writeDecodedToken(b *strings.Builder, token string) {
	if decoded, ok := decodeByteToken(token); ok {
		b.WriteByte(decoded)
		return
	}
	b.WriteString(strings.ReplaceAll(token, "▁", " "))
}

func decodeByteToken(token string) (byte, bool) {
	if len(token) != len("<0x00>") || !strings.HasPrefix(token, "<0x") || !strings.HasSuffix(token, ">") {
		return 0, false
	}
	value, err := strconv.ParseUint(token[3:5], 16, 8)
	if err != nil {
		return 0, false
	}
	return byte(value), true
}
