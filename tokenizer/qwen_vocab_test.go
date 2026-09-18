package tokenizer

import (
	"reflect"
	"testing"
)

func TestNewQwenCopiesVocabulary(t *testing.T) {
	options := QwenOptions{
		Tokens: []string{"a", "b", "ab", "<special>"}, Merges: []string{"a b"}, AddedTokenIDs: []int{3, 3},
	}
	tok, err := NewQwen(options)
	if err != nil {
		t.Fatal(err)
	}
	options.Tokens[2], options.Merges[0], options.AddedTokenIDs[0] = "changed", "x y", 0
	ids, err := tok.Encode("ab<special>ab")
	if err != nil || !reflect.DeepEqual(ids, []int{2, 3, 2}) {
		t.Fatalf("encode = %v, %v", ids, err)
	}
	decoded, err := tok.Decode(ids)
	if err != nil || decoded != "ab<special>ab" {
		t.Fatalf("decode = %q, %v", decoded, err)
	}
}

func TestNewQwenRejectsMalformedVocabulary(t *testing.T) {
	for _, options := range []QwenOptions{
		{},
		{Tokens: []string{""}},
		{Tokens: []string{"a", "a"}},
		{Tokens: []string{"a"}, AddedTokenIDs: []int{-1}},
		{Tokens: []string{"a"}, AddedTokenIDs: []int{1}},
		{Tokens: []string{"a"}, Merges: []string{"a"}},
		{Tokens: []string{"a", "b"}, Merges: []string{"a b"}},
		{Tokens: []string{"a", "ab"}, Merges: []string{"a b"}},
	} {
		if _, err := NewQwen(options); err == nil {
			t.Fatalf("expected error for %+v", options)
		}
	}
}
