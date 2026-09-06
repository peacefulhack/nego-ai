package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadEncodeDecodeAndCount(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
		"model": {
			"type": "WordLevel",
			"unk_token": "[UNK]",
			"vocab": {
				"[UNK]": 0,
				"hello": 1,
				"Ġworld": 2,
				"!": 3
			}
		}
	}`)

	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("hello world!")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, []int{1, 2, 3}) {
		t.Fatalf("ids = %#v", ids)
	}
	text, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello world!" {
		t.Fatalf("text = %q", text)
	}
	count, err := tok.Count("hello world!")
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count = %d", count)
	}
}

func TestLoadUsesAddedTokens(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
		"model": {
			"type": "WordLevel",
			"vocab": {"hello": 1}
		},
		"added_tokens": [{"id": 2, "content": "<|end|>"}]
	}`)

	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("hello<|end|>")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, []int{1, 2}) {
		t.Fatalf("ids = %#v", ids)
	}
}

func TestBPEUsesMergeRules(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
		"model": {
			"type": "BPE",
			"unk_token": "[UNK]",
			"vocab": {
				"[UNK]": 0,
				"l": 1,
				"o": 2,
				"w": 3,
				"e": 4,
				"r": 5,
				"lo": 6,
				"low": 7,
				"er": 8,
				"lower": 9
			},
			"merges": ["l o", "lo w", "e r"]
		}
	}`)

	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("lower")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, []int{7, 8}) {
		t.Fatalf("ids = %#v", ids)
	}
}

func TestLoadUnigramArrayVocab(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
		"model": {
			"type": "Unigram",
			"unk_id": 0,
			"vocab": [
				["<unk>", 0.0],
				["▁hello", -1.0],
				["▁world", -2.0],
				["!", -3.0]
			]
		}
	}`)

	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("hello world!")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, []int{1, 2, 3}) {
		t.Fatalf("ids = %#v", ids)
	}
	text, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if text != " hello world!" {
		t.Fatalf("text = %q", text)
	}
}

func TestBatchHelpers(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
		"model": {
			"type": "WordLevel",
			"unk_token": "[UNK]",
			"vocab": {
				"[UNK]": 0,
				"hello": 1,
				"bye": 2
			}
		}
	}`)

	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := tok.EncodeBatch([]string{"hello", "bye"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(encoded, [][]int{{1}, {2}}) {
		t.Fatalf("encoded = %#v", encoded)
	}
	decoded, err := tok.DecodeBatch(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, []string{"hello", "bye"}) {
		t.Fatalf("decoded = %#v", decoded)
	}
	counts, err := tok.CountBatch([]string{"hello", "bye"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(counts, []int{1, 1}) {
		t.Fatalf("counts = %#v", counts)
	}
}

func writeTokenizer(t *testing.T, dir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
