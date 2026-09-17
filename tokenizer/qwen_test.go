package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func qwenTestConfig(t *testing.T) map[string]any {
	t.Helper()
	vocab := make(map[string]int)
	for value, token := range byteLevelAlphabet {
		vocab[token] = value
	}
	// Deliberate cross-boundary merges must not combine digits or punctuation.
	merges := [][]string{{"1", "2"}, {"12", "3"}, {"a", "!"}, {"a", "b"}, {"a", "a"}, {"aa", "aa"}}
	for i, pair := range merges {
		vocab[pair[0]+pair[1]] = 256 + i
	}
	// A vocabulary entry without a merge must never shortcut BPE.
	vocab["\u0120a"] = 300
	return map[string]any{
		"model":      map[string]any{"type": "BPE", "vocab": vocab, "merges": merges},
		"normalizer": map[string]any{"type": "NFC"},
		"pre_tokenizer": map[string]any{"type": "Sequence", "pretokenizers": []any{
			map[string]any{"type": "Split", "pattern": map[string]string{"Regex": qwenPattern}, "behavior": "Isolated", "invert": false},
			map[string]any{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
		}},
		"decoder": map[string]any{"type": "ByteLevel"},
		"added_tokens": []any{
			map[string]any{"id": 400, "content": "<x>"},
			map[string]any{"id": 401, "content": "<x>long"},
			map[string]any{"id": 402, "content": "<y>"},
			map[string]any{"id": 403, "content": "two words"},
		},
	}
}

func loadQwenTestConfig(t *testing.T, config map[string]any) (*Tokenizer, error) {
	t.Helper()
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	writeTokenizer(t, dir, string(data))
	return Load(dir)
}

func TestQwenEncodePipeline(t *testing.T) {
	tok, err := loadQwenTestConfig(t, qwenTestConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		text    string
		want    []int
		decoded string
	}{
		{"123", []int{49, 50, 51}, "123"},
		{"ab!", []int{259, 33}, "ab!"},
		{"a!", []int{97, 33}, "a!"},
		{"aaaa", []int{261}, "aaaa"},
		{" a", []int{32, 97}, " a"},
		{"\u00e9", []int{195, 169}, "\u00e9"},
		{"e\u0301", []int{195, 169}, "\u00e9"},
		{"\U0001f600", []int{240, 159, 152, 128}, "\U0001f600"},
		{"<x>long<y><x>", []int{401, 402, 400}, "<x>long<y><x>"},
		{"a<x>b<y>", []int{97, 400, 98, 402}, "a<x>b<y>"},
		{"two words", []int{403}, "two words"},
		{"", nil, ""},
	} {
		t.Run(tc.text, func(t *testing.T) {
			ids, err := tok.Encode(tc.text)
			if err != nil || !reflect.DeepEqual(ids, tc.want) {
				t.Fatalf("Encode(%q) = %v, %v; want %v", tc.text, ids, err, tc.want)
			}
			decoded, err := tok.Decode(ids)
			if err != nil || decoded != tc.decoded {
				t.Fatalf("Decode = %q, %v; want %q", decoded, err, tc.decoded)
			}
		})
	}
	if _, err := tok.Encode(string([]byte{0xff})); err == nil {
		t.Fatal("invalid UTF-8 should fail")
	}
	if _, err := tok.Encode("a" + strings.Repeat("\u0301", 40)); err == nil || !strings.Contains(err.Error(), "combining-sequence") {
		t.Fatalf("expected long combining sequence error, got %v", err)
	}
}

func TestQwenConfigRejectsUnsupportedOptions(t *testing.T) {
	for _, change := range []func(map[string]any){
		func(c map[string]any) { c["decoder"] = map[string]any{"type": "WordPiece"} },
		func(c map[string]any) { c["normalizer"] = map[string]any{"type": "NFKC"} },
		func(c map[string]any) { c["model"].(map[string]any)["dropout"] = 0.1 },
		func(c map[string]any) { c["model"].(map[string]any)["ignore_merges"] = true },
		func(c map[string]any) { c["added_tokens"].([]any)[0].(map[string]any)["single_word"] = true },
		func(c map[string]any) {
			c["pre_tokenizer"].(map[string]any)["pretokenizers"].([]any)[1].(map[string]any)["use_regex"] = true
		},
		func(c map[string]any) {
			c["pre_tokenizer"].(map[string]any)["pretokenizers"].([]any)[0].(map[string]any)["pattern"] = map[string]string{"Regex": ".*"}
		},
	} {
		config := qwenTestConfig(t)
		change(config)
		if _, err := loadQwenTestConfig(t, config); err == nil || !strings.Contains(err.Error(), "unsupported") {
			t.Fatalf("expected unsupported config error, got %v", err)
		}
	}
}

type qwenReferenceCase struct {
	Name     string   `json:"name"`
	Text     string   `json:"text"`
	Segments []string `json:"segments"`
	Tokens   []int    `json:"tokens"`
	Decoded  string   `json:"decoded"`
}

func readQwenReference(t *testing.T) []qwenReferenceCase {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "qwen_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []qwenReferenceCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("reference fixture has no cases")
	}
	return fixture.Cases
}

func TestQwenWithoutNormalizer(t *testing.T) {
	config := qwenTestConfig(t)
	delete(config, "normalizer")
	tok, err := loadQwenTestConfig(t, config)
	if err != nil {
		t.Fatal(err)
	}
	got, err := tok.Encode("e\u0301")
	if err != nil || !reflect.DeepEqual(got, []int{101, 204, 129}) {
		t.Fatalf("encoding without NFC = %v, %v", got, err)
	}
}

func TestQwenSplitMatchesReference(t *testing.T) {
	for _, tc := range readQwenReference(t) {
		t.Run(tc.Name, func(t *testing.T) {
			got := splitQwen(tc.Text)
			if strings.Join(got, "") != tc.Text || len(got) != len(tc.Segments) {
				t.Fatalf("split = %q, want %q", got, tc.Segments)
			}
			for i := range got {
				if got[i] != tc.Segments[i] {
					t.Fatalf("split = %q, want %q", got, tc.Segments)
				}
			}
		})
	}
}

func TestQwenLocalReferenceTokenIDs(t *testing.T) {
	path := os.Getenv("NEGO_QWEN_TOKENIZER_DIR")
	if path == "" {
		t.Skip("set NEGO_QWEN_TOKENIZER_DIR to a local Qwen3 tokenizer directory")
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !tok.qwenEncoder {
		t.Fatal("local tokenizer does not use the Qwen pipeline")
	}
	for _, tc := range readQwenReference(t) {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := tok.Encode(tc.Text)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.Tokens) || len(got) > 0 && !reflect.DeepEqual(got, tc.Tokens) {
				t.Fatalf("IDs = %v, want %v", got, tc.Tokens)
			}
			decoded, err := tok.Decode(got)
			if err != nil || decoded != tc.Decoded {
				t.Fatalf("Decode = %q, %v; want %q", decoded, err, tc.Decoded)
			}
		})
	}
}
