package patterns

import "testing"

func TestMatchIncludeMatchesNestedPathsLikeHF(t *testing.T) {
	if !Match("onnx/config.json", []string{"*.json"}, nil) {
		t.Fatal("expected *.json to match nested JSON files")
	}
}

func TestMatchExcludeWins(t *testing.T) {
	if Match("model.msgpack", []string{"*"}, []string{"*.msgpack"}) {
		t.Fatal("expected exclude pattern to win")
	}
}
