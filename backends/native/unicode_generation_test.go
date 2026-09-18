package native

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

func TestGGUFGenerationStreamsCompleteUnicode(t *testing.T) {
	for _, tc := range []struct {
		name      string
		maxTokens int
		eos       bool
		want      string
	}{
		{"complete", 2, false, "\u00e9"},
		{"token limit flush", 1, false, "\ufffd"},
		{"eos flush", 2, true, "\ufffd"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			loaded, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: fakeBlockGGUF(t)})
			if err != nil {
				t.Fatal(err)
			}
			defer loaded.Close()
			m := loaded.(*Model)
			m.vocab = &modelinfo.GGUFVocab{Model: "gpt2", PreTokenizer: "qwen2", Tokens: []string{"p", "\u00c3", "\u00a9", "x", "y", "z"}}
			if tc.eos {
				m.vocab.EOSTokenID = 2
			}
			// Zero residual branches preserve the embedding direction. Output
			// weights alternate the C3/A9 bytes of e-acute after prompt "p".
			for _, name := range []string{m.names.Blocks[0].AttentionOut, m.names.Blocks[0].FFNDown} {
				values, tensor, err := m.loadTensorFloat32Shared(name)
				if err != nil {
					t.Fatal(err)
				}
				m.float32[name] = cachedFloat32Tensor{tensor: tensor, values: make([]float32, len(values))}
			}
			for name, values := range map[string][]float32{
				m.names.TokenEmbedding: {1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0},
				m.names.Output:         {0, 0, 1, 0, 0, 1, -1, -1, -1, -1, -1, -1},
			} {
				_, tensor, err := m.loadTensorFloat32Shared(name)
				if err != nil {
					t.Fatal(err)
				}
				m.float32[name] = cachedFloat32Tensor{tensor: tensor, values: values}
			}
			req := nego.GenerateRequest{Prompt: "p", MaxTokens: tc.maxTokens}
			out, err := m.Generate(context.Background(), req)
			if err != nil || out.Text != tc.want {
				t.Fatalf("generate = %v, %v; want %q", out, err, tc.want)
			}
			stream, err := m.StreamGenerate(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			var text strings.Builder
			for token := range stream.Tokens() {
				if !utf8.ValidString(token.Text) {
					t.Fatalf("invalid UTF-8 chunk %q", token.Text)
				}
				text.WriteString(token.Text)
			}
			if err := stream.Err(); err != nil {
				t.Fatal(err)
			}
			if text.String() != tc.want {
				t.Fatalf("stream = %q, want %q", text.String(), tc.want)
			}
		})
	}
}
