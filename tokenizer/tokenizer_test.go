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

func writeTokenizer(t *testing.T, dir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
