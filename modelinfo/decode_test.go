package modelinfo

import (
	"strings"
	"testing"
)

func TestGGUFVocabDecode(t *testing.T) {
	vocab := &GGUFVocab{
		Tokens:     []string{"<s>", "▁Hello", "▁world", "!", "<0x0A>"},
		TokenTypes: []uint32{3, 1, 1, 1, 1},
		BOSTokenID: 0,
		EOSTokenID: 0,
	}
	got, err := vocab.Decode([]int{0, 1, 2, 3, 4}, DecodeOptions{SkipSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != " Hello world!\n" {
		t.Fatalf("got %q", got)
	}
}

func TestGGUFVocabDecodeRejectsInvalidID(t *testing.T) {
	vocab := &GGUFVocab{Tokens: []string{"hello"}}
	_, err := vocab.Decode([]int{1}, DecodeOptions{})
	if err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGGUFVocabDecodeSkipsKnownSpecialIDZero(t *testing.T) {
	vocab := &GGUFVocab{Tokens: []string{"<s>", "hello"}}
	got, err := vocab.Decode([]int{0, 1}, DecodeOptions{SkipSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello" {
		t.Fatalf("Decode = %q, want hello", got)
	}
}

func TestGGUFVocabDecodeRejectsNilVocab(t *testing.T) {
	_, err := (*GGUFVocab)(nil).Decode([]int{0}, DecodeOptions{})
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}
