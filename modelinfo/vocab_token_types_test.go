package modelinfo

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"strings"
	"testing"
)

func TestReadGGUFVocabTokenTypeEncodings(t *testing.T) {
	for _, tc := range []struct {
		name     string
		elemType uint32
		length   uint64
		values   any
		want     []uint32
		wantErr  string
	}{
		{"signed", 5, 3, []int32{3, 1, 4}, []uint32{3, 1, 4}, ""},
		{"unsigned", 4, 3, []uint32{3, 1, 4}, []uint32{3, 1, 4}, ""},
		{"negative", 5, 2, []int32{1, -1}, nil, "item 1 has negative token type"},
		{"truncated", 5, 2, []int32{1}, nil, "item 1"},
		{"wrong type", 6, 1, []float32{1}, nil, "want int32 or uint32"},
		{"length limit", 5, maxGGUFVocabTokens + 1, []int32{}, nil, "exceeds limit"},
		{"empty", 5, 0, []int32{}, []uint32{}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			writeGGUFHeader(t, &buf, 3, 0, 1)
			writeGGUFString(t, &buf, "tokenizer.ggml.token_type")
			for _, value := range []any{uint32(9), tc.elemType, tc.length, tc.values} {
				if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
					t.Fatal(err)
				}
			}
			vocab, err := ReadGGUFVocab(bytes.NewReader(buf.Bytes()), "fixture.gguf")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) || vocab != nil {
					t.Fatalf("got vocab=%v error=%v; want %q", vocab, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(vocab.TokenTypes, tc.want) {
				t.Fatalf("types=%v, want %v", vocab.TokenTypes, tc.want)
			}
		})
	}
}
