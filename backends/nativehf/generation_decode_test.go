package nativehf

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	nego "github.com/gakon/nego-ai"
)

func TestGenerationAssemblesByteLevelTokens(t *testing.T) {
	dir := writeTinyForwardModel(t)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{
		"model":{"type":"BPE","vocab":{"\u00c3":0,"\u00a9":1}},
		"decoder":{"type":"ByteLevel"}
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	loaded, err := (Backend{}).Load(context.Background(), nego.ModelOptions{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	model := loaded.(*Model)
	// Isolate the residual embedding and swap the output classes so generation
	// alternates byte tokens C3/A9, forming U+00E9 every two generated IDs.
	for name, values := range map[string][]float32{
		"lm_head.weight":                         {0, 1, 1, 0},
		"model.layers.0.self_attn.o_proj.weight": {0, 0, 0, 0},
	} {
		_, tensor, err := model.LoadTensorFloat32(name)
		if err != nil {
			t.Fatal(err)
		}
		model.float32[name] = cachedFloat32Tensor{values: values, tensor: tensor}
	}
	for _, tc := range []struct {
		name string
		max  int
		stop []string
		want string
	}{
		{"complete", 4, nil, "\u00e9\u00e9"},
		{"incomplete at limit", 1, nil, "\ufffd"},
		{"unicode stop", 4, []string{"\u00e9"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := nego.GenerateRequest{Prompt: "\u00a9", MaxTokens: tc.max, Stop: tc.stop}
			out, err := model.Generate(context.Background(), req)
			if err != nil || out.Text != tc.want {
				t.Fatalf("Generate = %v, %v; want %q", out, err, tc.want)
			}
			stream, err := model.StreamGenerate(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			var text strings.Builder
			for token := range stream.Tokens() {
				if !utf8.ValidString(token.Text) {
					t.Errorf("invalid UTF-8 streamed: %q", token.Text)
				}
				text.WriteString(token.Text)
			}
			if err := stream.Err(); err != nil || text.String() != tc.want {
				t.Fatalf("StreamGenerate = %q, %v; want %q", text.String(), err, tc.want)
			}
		})
	}
}
