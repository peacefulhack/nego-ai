package datasets

import (
	"bytes"
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
