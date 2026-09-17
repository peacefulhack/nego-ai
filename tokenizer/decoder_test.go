package tokenizer

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestByteLevelDecodeMergedTokens(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
		"model":{"type":"BPE","vocab":{
			"\u0120caf\u00c3\u00a9":0,"\u010a\u010a":1,
			"A\u0109B\u0120\u0120C":2,"<|im_end|>":3,
			"\u00e4\u00bd\u0142\u00e5\u00a5\u00bd":4,
			"\u00f0\u0141\u013a\u0122":5,"literal\u6c49":6
		}},"decoder":{"type":"ByteLevel"}
	}`)
	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		ids  []int
		want string
	}{
		{[]int{0, 1, 2, 3}, " caf\u00e9\n\nA\tB  C<|im_end|>"},
		{[]int{4}, "\u4f60\u597d"},
		{[]int{5}, "\U0001f600"},
		{[]int{6}, "literal\u6c49"},
		{nil, ""},
	} {
		got, err := tok.Decode(tc.ids)
		if err != nil || got != tc.want {
			t.Fatalf("Decode(%v) = %q, %v; want %q", tc.ids, got, err, tc.want)
		}
	}
	if _, err := tok.Decode([]int{999}); err == nil {
		t.Fatal("expected unknown token error")
	}
}

func TestByteLevelStreamingDecode(t *testing.T) {
	tok := &Tokenizer{
		byteLevelDecoder: true,
		IDToToken: map[int]string{
			0: "\u00f0", 1: "\u0141", 2: "\u013a", 3: "\u0122", // F0 9F 98 80
			4: "!", 5: "\u00e2\u0124", // Incomplete E2 82
		},
	}
	decoder := tok.NewDecoder()
	var streamed strings.Builder
	for i, want := range []string{"", "", "", "\U0001f600", "!"} {
		got, err := decoder.Push(i)
		if err != nil || got != want || !utf8.ValidString(got) {
			t.Fatalf("Push(%d) = %q, %v; want %q", i, got, err, want)
		}
		streamed.WriteString(got)
	}
	if got := decoder.Flush(); got != "" {
		t.Fatalf("unexpected pending bytes: %q", got)
	}
	decoded, err := tok.Decode([]int{0, 1, 2, 3, 4})
	if err != nil || decoded != streamed.String() {
		t.Fatalf("full and streamed decode differ: %q, %v", decoded, err)
	}
	if got, err := decoder.Push(5); got != "" || err != nil {
		t.Fatalf("incomplete token = %q, %v", got, err)
	}
	if got := decoder.Flush(); got != "\ufffd" {
		t.Fatalf("incomplete final bytes = %q", got)
	}
	if got := decoder.Flush(); got != "" {
		t.Fatalf("second Flush = %q", got)
	}
}

func TestByteLevelDecoderUnknownIDPreservesPending(t *testing.T) {
	tok := &Tokenizer{byteLevelDecoder: true, IDToToken: map[int]string{0: "\u00c3", 1: "\u00a9"}}
	decoder := tok.NewDecoder()
	if _, err := decoder.Push(0); err != nil {
		t.Fatal(err)
	}
	if _, err := decoder.Push(123); err == nil {
		t.Fatal("expected unknown token error")
	}
	got, err := decoder.Push(1)
	if err != nil || got != "\u00e9" {
		t.Fatalf("pending UTF-8 was lost: %q, %v", got, err)
	}
	var uninitialized Decoder
	if _, err := uninitialized.Push(0); err == nil {
		t.Fatal("expected uninitialized decoder error")
	}
}

func TestDecoderKeepsLegacyNonByteLevelBehavior(t *testing.T) {
	tok := &Tokenizer{IDToToken: map[int]string{0: "caf\u00e9", 1: "\u0120world", 2: "\u010a"}}
	got, err := tok.Decode([]int{0, 1, 2})
	if err != nil || got != "caf\u00e9 world\n" {
		t.Fatalf("legacy decode = %q, %v", got, err)
	}
}

func TestByteLevelAlphabet(t *testing.T) {
	if len(byteLevelReverse) != 256 {
		t.Fatalf("alphabet has %d entries", len(byteLevelReverse))
	}
	seen := make(map[byte]bool)
	for _, value := range byteLevelReverse {
		if seen[value] {
			t.Fatalf("duplicate byte %d", value)
		}
		seen[value] = true
	}
	for char, want := range map[rune]byte{'!': 33, '~': 126, '\u00a1': 161, '\u00ff': 255, '\u0100': 0, '\u0109': 9, '\u010a': 10, '\u0120': 32, '\u0143': 173} {
		if got, ok := byteLevelByte(char); !ok || got != want {
			t.Fatalf("alphabet %U = %d, %v; want %d", char, got, ok, want)
		}
	}
}
