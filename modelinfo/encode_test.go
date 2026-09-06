package modelinfo

import (
	"strings"
	"testing"
)

func TestGGUFVocabEncode(t *testing.T) {
	vocab := &GGUFVocab{
		Tokens:     []string{"<unk>", "<s>", "</s>", "hello", "▁world", "!"},
		UNKTokenID: 0,
		BOSTokenID: 1,
		EOSTokenID: 2,
	}
	got, err := vocab.Encode("hello world!", EncodeOptions{AddBOS: true, AddEOS: true})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 3, 4, 5, 2}
	assertIDs(t, got, want)
}

func TestGGUFVocabEncodeByteFallback(t *testing.T) {
	vocab := &GGUFVocab{Tokens: []string{"<unk>", "<0x0A>"}}
	got, err := vocab.Encode("\n", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, got, []int{1})
}

func TestGGUFVocabEncodeUsesBPEMerges(t *testing.T) {
	vocab := &GGUFVocab{
		Tokens: []string{"<unk>", "l", "o", "w", "e", "r", "lo", "low", "er", "lower"},
		Merges: []string{"l o", "lo w", "e r"},
	}
	got, err := vocab.Encode("lower", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, got, []int{7, 8})
}

func TestGGUFVocabEncodePrefersExactTokenWithMerges(t *testing.T) {
	vocab := &GGUFVocab{
		Tokens: []string{"<unk>", "hello"},
		Merges: []string{"h e"},
	}
	got, err := vocab.Encode("hello", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, got, []int{1})
}

func TestGGUFVocabEncodeUsesZeroUnknownToken(t *testing.T) {
	vocab := &GGUFVocab{Tokens: []string{"<unk>"}}
	got, err := vocab.Encode("?", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, got, []int{0})
}

func TestGGUFVocabEncodeRejectsUnknown(t *testing.T) {
	vocab := &GGUFVocab{Tokens: []string{"hello"}}
	_, err := vocab.Encode("bye", EncodeOptions{})
	if err == nil || !strings.Contains(err.Error(), "cannot encode") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGGUFVocabEncodeRejectsNil(t *testing.T) {
	_, err := (*GGUFVocab)(nil).Encode("hello", EncodeOptions{})
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func assertIDs(t *testing.T, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("ids = %#v, want %#v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("ids = %#v, want %#v", got, want)
		}
	}
}
