package modelinfo

import (
	"fmt"

	"github.com/gakon/nego-ai/tokenizer"
)

func (v *GGUFVocab) qwenTokenizer() (*tokenizer.Tokenizer, error) {
	if v.Model != "gpt2" {
		return nil, fmt.Errorf("GGUF qwen2 pre-tokenizer requires gpt2 vocabulary, got %q", v.Model)
	}
	if v.AddSpace || v.RemoveSpaces {
		return nil, fmt.Errorf("GGUF qwen2 whitespace normalization options are not supported")
	}
	if len(v.TokenTypes) != 0 && len(v.TokenTypes) != len(v.Tokens) {
		return nil, fmt.Errorf("GGUF token type count does not match vocabulary size")
	}
	var added []int
	for id := range v.Tokens {
		if v.isSpecialToken(id) || id < len(v.TokenTypes) && v.TokenTypes[id] == 4 {
			added = append(added, id)
		}
	}
	return tokenizer.NewQwen(tokenizer.QwenOptions{Tokens: v.Tokens, Merges: v.Merges, AddedTokenIDs: added})
}

func (v *GGUFVocab) encodeQwen(text string, options EncodeOptions) ([]int, error) {
	tok, err := v.qwenTokenizer()
	if err != nil {
		return nil, err
	}
	ids, err := tok.Encode(text)
	if err != nil {
		return nil, err
	}
	if options.AddBOS {
		if uint64(v.BOSTokenID) >= uint64(len(v.Tokens)) {
			return nil, fmt.Errorf("GGUF BOS token id is out of range")
		}
		ids = append([]int{int(v.BOSTokenID)}, ids...)
	}
	if options.AddEOS {
		if uint64(v.EOSTokenID) >= uint64(len(v.Tokens)) {
			return nil, fmt.Errorf("GGUF EOS token id is out of range")
		}
		ids = append(ids, int(v.EOSTokenID))
	}
	return ids, nil
}
