package datasets

import (
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
