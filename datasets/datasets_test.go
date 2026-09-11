package datasets

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadJSONL(t *testing.T) {
	rows, err := ReadJSONL(strings.NewReader("{\"prompt\":\"hi\"}\n{\"prompt\":\"bye\"}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0]["prompt"] != "hi" {
		t.Fatalf("rows = %#v", rows)
	}
}

func TestReadCSV(t *testing.T) {
	rows, err := ReadCSV(strings.NewReader("prompt,response\nhi,hello\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0]["response"] != "hello" {
		t.Fatalf("rows = %#v", rows)
	}
}

func TestSplit(t *testing.T) {
	rows := []Row{{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}}
	train, test := Split(rows, 0.25, 42)
	if len(train) != 3 || len(test) != 1 {
		t.Fatalf("train=%d test=%d", len(train), len(test))
	}
}

func TestSelectAndRequireFields(t *testing.T) {
	rows := []Row{
		{"prompt": "hi", "completion": "hello", "meta": "keep out"},
		{"prompt": "bye", "completion": "", "meta": "drop"},
	}
	required := RequireFields(rows, []string{"prompt", "completion"})
	if len(required) != 1 || required[0]["prompt"] != "hi" {
		t.Fatalf("required = %#v", required)
	}
	selected := SelectFields(required, []string{"prompt", "completion"})
	if _, ok := selected[0]["meta"]; ok {
		t.Fatalf("unexpected meta field: %#v", selected[0])
	}
}

func TestParseAndFilterRows(t *testing.T) {
	rows := []Row{
		{"split": "train", "prompt": "hello world"},
		{"split": "test", "prompt": "bye"},
		{"split": "train", "prompt": "skip this"},
	}
	eq, err := ParseFilter("split=train")
	if err != nil {
		t.Fatal(err)
	}
	notContains, err := ParseFilter("prompt!~skip")
	if err != nil {
		t.Fatal(err)
	}
	filtered := FilterRows(rows, []Filter{eq, notContains})
	if len(filtered) != 1 || filtered[0]["prompt"] != "hello world" {
		t.Fatalf("filtered = %#v", filtered)
	}
	exists, err := ParseFilter("prompt")
	if err != nil {
		t.Fatal(err)
	}
	if len(FilterRows(rows, []Filter{exists})) != 3 {
		t.Fatalf("exists filter did not match all prompt rows")
	}
}

func TestValidateRequiredFields(t *testing.T) {
	if err := ValidateRequiredFields([]Row{{"prompt": "hi"}}, "prompt"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateRequiredFields([]Row{{"prompt": "hi"}}, "response"); err == nil {
		t.Fatal("expected missing field error")
	}
}

func TestWriteJSONL(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteJSONL(&buf, []Row{{"prompt": "hi"}, {"prompt": "bye"}}); err != nil {
		t.Fatal(err)
	}
	rows, err := ReadJSONL(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[1]["prompt"] != "bye" {
		t.Fatalf("rows = %#v", rows)
	}
}

func TestValidateFormat(t *testing.T) {
	chat := []Row{{"messages": []any{
		map[string]any{"role": "user", "content": "hi"},
		map[string]any{"role": "assistant", "content": "hello"},
	}}}
	if err := ValidateFormat(chat, "chat"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateFormat([]Row{{"prompt": "hi", "completion": "hello"}}, "completion"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateFormat([]Row{{"instruction": "say hi", "output": "hi"}}, "instruction"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateFormat([]Row{{"messages": []any{map[string]any{"role": "tool", "content": "bad"}}}}, "chat"); err == nil {
		t.Fatal("expected invalid role error")
	}
}

func TestAnalyzeTokenBudget(t *testing.T) {
	dir := t.TempDir()
	writeDatasetTokenizer(t, dir)
	rows := []Row{
		{"prompt": "hello", "completion": "world"},
		{"prompt": "hello hello hello", "completion": "world"},
	}
	report, err := AnalyzeTokenBudget(rows, TokenBudgetOptions{
		ModelPath:  dir,
		Format:     "completion",
		MaxContext: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Valid || report.OverLimit != 1 || report.MaxTokens <= report.MinTokens {
		t.Fatalf("unexpected report: %#v", report)
	}
	longest := LongestTokenRows(report.RowsDetail, 1)
	if len(longest) != 1 || longest[0].Row != 2 {
		t.Fatalf("unexpected longest rows: %#v", longest)
	}
}

func TestAnalyzeTokenBudgetChatRows(t *testing.T) {
	dir := t.TempDir()
	writeDatasetTokenizer(t, dir)
	rows := []Row{{"messages": []any{
		map[string]any{"role": "user", "content": "hello"},
		map[string]any{"role": "assistant", "content": "world"},
	}}}
	report, err := AnalyzeTokenBudget(rows, TokenBudgetOptions{ModelPath: dir, Format: "chat"})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Valid || report.RowsDetail[0].Format != "chat" || report.RowsDetail[0].Tokens == 0 {
		t.Fatalf("unexpected report: %#v", report)
	}
}

func writeDatasetTokenizer(t *testing.T, dir string) {
	t.Helper()
	body := `{"model":{"type":"WordLevel","unk_token":"[UNK]","vocab":{"[UNK]":0,"hello":1,"world":2,"Ġhello":3,"Ġworld":4,"Instruction":5,"Output":6,":":7,"Ċ":8}}}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
