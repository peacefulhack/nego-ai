package modelinfo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func qwenTestVocab() *GGUFVocab {
	return &GGUFVocab{
		Model: "gpt2", PreTokenizer: "qwen2",
		Tokens:     []string{"h", "i", "hi", "\u0120", "\u010a", "\u00c3", "\u00a9", "<|im_start|>", "<|im_end|>", "e", "\u00cc", "\u0123"},
		TokenTypes: []uint32{1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1},
		Merges:     []string{"h i"}, BOSTokenID: 7, EOSTokenID: 8,
	}
}

func TestQwenGGUFReadinessTokenizerWarning(t *testing.T) {
	info := &Info{GGUF: &GGUFInfo{Metadata: map[string]any{
		"tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.model": "gpt2",
	}}}
	warnings := strings.Join(nativeReadinessWarnings(info, &NativeRuntimeReadiness{}), "\n")
	if !strings.Contains(warnings, "without HF NFC") || strings.Contains(warnings, "approximate") {
		t.Fatalf("unexpected Qwen warning: %q", warnings)
	}
}

func TestGGUFQwenEncodeDecode(t *testing.T) {
	v := qwenTestVocab()
	text := " hi\n\u00e9<|im_start|>hi<|im_end|>"
	ids, err := v.Encode(text, EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, ids, []int{3, 2, 4, 5, 6, 7, 2, 8})
	decoded, err := v.Decode(ids, DecodeOptions{})
	if err != nil || decoded != text {
		t.Fatalf("decode = %q, %v", decoded, err)
	}
	decoded, err = v.Decode(ids, DecodeOptions{SkipSpecial: true})
	if err != nil || decoded != " hi\n\u00e9hi" {
		t.Fatalf("skip special = %q, %v", decoded, err)
	}
	ids, err = v.Encode("hi", EncodeOptions{AddBOS: true, AddEOS: true})
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, ids, []int{7, 2, 8})
	ids, err = v.Encode("e\u0301", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, ids, []int{9, 10, 11}) // GGUF does not implicitly apply HF NFC.
}

func TestGGUFQwenIncrementalDecoder(t *testing.T) {
	v := qwenTestVocab()
	d, err := v.NewDecoder(DecodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if text, err := d.Push(5); err != nil || text != "" {
		t.Fatalf("partial byte = %q, %v", text, err)
	}
	if _, err := d.Push(-1); err == nil {
		t.Fatal("expected invalid token error")
	}
	if text, err := d.Push(6); err != nil || text != "\u00e9" || !utf8.ValidString(text) {
		t.Fatalf("completed character = %q, %v", text, err)
	}
	if _, err := d.Push(5); err != nil {
		t.Fatal(err)
	}
	if text := d.Flush(); text != "\ufffd" || d.Flush() != "" {
		t.Fatalf("flush = %q", text)
	}
	d2, err := v.NewDecoder(DecodeOptions{})
	if err != nil || d2.Flush() != "" {
		t.Fatal("new decoder must have independent state")
	}
}

func TestGGUFQwenRejectsUnsupportedOptions(t *testing.T) {
	for _, mutate := range []func(*GGUFVocab){
		func(v *GGUFVocab) { v.Model = "llama" },
		func(v *GGUFVocab) { v.AddSpace = true },
		func(v *GGUFVocab) { v.RemoveSpaces = true },
		func(v *GGUFVocab) { v.TokenTypes = []uint32{1} },
	} {
		v := qwenTestVocab()
		mutate(v)
		if _, err := v.Encode("hi", EncodeOptions{}); err == nil {
			t.Fatal("expected unsupported metadata error")
		}
		if _, err := v.NewDecoder(DecodeOptions{}); err == nil {
			t.Fatal("decoder must reject unsupported metadata too")
		}
	}
	v := qwenTestVocab()
	v.BOSTokenID, v.EOSTokenID = 999, 999
	for _, options := range []EncodeOptions{{AddBOS: true}, {AddEOS: true}} {
		if _, err := v.Encode("hi", options); err == nil {
			t.Fatal("expected invalid special token error")
		}
	}
}

func TestQwenGGUFLocalReference(t *testing.T) {
	path := os.Getenv("NEGO_QWEN_GGUF")
	if path == "" {
		t.Skip("set NEGO_QWEN_GGUF to a local Qwen3-0.6B GGUF file")
	}
	v, err := InspectGGUFVocab(path)
	if err != nil {
		t.Fatal(err)
	}
	if v.PreTokenizer != "qwen2" {
		t.Fatalf("unexpected pre-tokenizer %q", v.PreTokenizer)
	}
	data, err := os.ReadFile(filepath.Join("..", "tokenizer", "testdata", "qwen_gguf_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name, Text, Decoded string
			Tokens              []int
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 24 {
		t.Fatalf("expected 24 reference cases, got %d", len(fixture.Cases))
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			ids, err := v.Encode(tc.Text, EncodeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			assertIDs(t, ids, tc.Tokens)
			got, err := v.Decode(ids, DecodeOptions{})
			if err != nil || got != tc.Decoded {
				t.Fatalf("decode = %q, %v; want %q", got, err, tc.Decoded)
			}
			d, err := v.NewDecoder(DecodeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			var out strings.Builder
			for _, id := range ids {
				part, err := d.Push(id)
				if err != nil || !utf8.ValidString(part) {
					t.Fatalf("chunk = %q, %v", part, err)
				}
				out.WriteString(part)
			}
			out.WriteString(d.Flush())
			if out.String() != got {
				t.Fatalf("stream = %q, want %q", out.String(), got)
			}
		})
	}
}
