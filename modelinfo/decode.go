package modelinfo

import (
	"fmt"
	"strconv"
	"strings"
)

type DecodeOptions struct {
	SkipSpecial bool
}

func (v *GGUFVocab) Decode(ids []int, options DecodeOptions) (string, error) {
	if v == nil {
		return "", fmt.Errorf("gguf vocab is nil")
	}
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(v.Tokens) {
			return "", fmt.Errorf("token id %d is out of range", id)
		}
		if options.SkipSpecial && v.isSpecialToken(id) {
			continue
		}
		writeDecodedToken(&b, v.Tokens[id])
	}
	return b.String(), nil
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
	if id < len(v.TokenTypes) && v.TokenTypes[id] == 3 {
		return true
	}
	return false
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
