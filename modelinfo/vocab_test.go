package modelinfo

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func TestReadGGUFVocabParsesTokenizerMetadata(t *testing.T) {
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 0, 8)
	writeGGUFStringKV(t, &buf, "tokenizer.ggml.model", "llama")
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello", "world"})
	writeGGUFFloat32ArrayKV(t, &buf, "tokenizer.ggml.scores", []float32{0, -1, -2})
	writeGGUFUint32ArrayKV(t, &buf, "tokenizer.ggml.token_type", []uint32{3, 1, 1})
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.merges", []string{"h e", "he llo"})
	writeGGUFUint32KV(t, &buf, "tokenizer.ggml.bos_token_id", 1)
	writeGGUFUint32KV(t, &buf, "tokenizer.ggml.eos_token_id", 2)
	writeGGUFStringKV(t, &buf, "tokenizer.chat_template", "{{ .Prompt }}")

	vocab, err := ReadGGUFVocab(bytes.NewReader(buf.Bytes()), "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if vocab.Model != "llama" || vocab.BOSTokenID != 1 || vocab.EOSTokenID != 2 {
		t.Fatalf("unexpected vocab metadata: %#v", vocab)
	}
	if strings.Join(vocab.Tokens, ",") != "<unk>,hello,world" {
		t.Fatalf("unexpected tokens: %#v", vocab.Tokens)
	}
	if len(vocab.Scores) != 3 || vocab.Scores[2] != -2 {
		t.Fatalf("unexpected scores: %#v", vocab.Scores)
	}
	if len(vocab.TokenTypes) != 3 || vocab.TokenTypes[0] != 3 {
		t.Fatalf("unexpected token types: %#v", vocab.TokenTypes)
	}
	if strings.Join(vocab.Merges, ",") != "h e,he llo" {
		t.Fatalf("unexpected merges: %#v", vocab.Merges)
	}
	if vocab.ChatTemplate != "{{ .Prompt }}" {
		t.Fatalf("unexpected chat template: %q", vocab.ChatTemplate)
	}
}

func TestReadGGUFVocabKeepsLargeTokenArray(t *testing.T) {
	values := make([]string, maxGGUFArrayStored+2)
	for i := range values {
		values[i] = "token"
	}
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 0, 1)
	writeGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", values)

	vocab, err := ReadGGUFVocab(bytes.NewReader(buf.Bytes()), "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if len(vocab.Tokens) != len(values) {
		t.Fatalf("tokens = %d, want %d", len(vocab.Tokens), len(values))
	}
}

func TestReadGGUFVocabRejectsWrongArrayType(t *testing.T) {
	var buf bytes.Buffer
	writeGGUFHeader(t, &buf, 3, 0, 1)
	writeGGUFFloat32ArrayKV(t, &buf, "tokenizer.ggml.tokens", []float32{1})

	_, err := ReadGGUFVocab(bytes.NewReader(buf.Bytes()), "model.gguf")
	if err == nil || !strings.Contains(err.Error(), "want string") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func writeGGUFFloat32ArrayKV(t *testing.T, buf *bytes.Buffer, key string, values []float32) {
	t.Helper()
	writeGGUFString(t, buf, key)
	for _, value := range []any{uint32(9), uint32(6), uint64(len(values))} {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range values {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
}

func writeGGUFUint32ArrayKV(t *testing.T, buf *bytes.Buffer, key string, values []uint32) {
	t.Helper()
	writeGGUFString(t, buf, key)
	for _, value := range []any{uint32(9), uint32(4), uint64(len(values))} {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range values {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
}
