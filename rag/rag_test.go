package rag

import (
	"strings"
	"testing"
)

func TestChunkText(t *testing.T) {
	chunks := ChunkText("abcdefghijklmnopqrstuvwxyz", ChunkOptions{MaxRunes: 10, Overlap: 2})
	if len(chunks) != 3 {
		t.Fatalf("chunks = %#v", chunks)
	}
	if chunks[0].Text != "abcdefghij" {
		t.Fatalf("first chunk = %q", chunks[0].Text)
	}
	if !strings.HasPrefix(chunks[1].Text, "ij") {
		t.Fatalf("expected overlap in second chunk, got %q", chunks[1].Text)
	}
}

func TestIndexSearch(t *testing.T) {
	index := NewIndex()
	index.Add(Document{ID: "a", Text: "go", Embedding: []float64{1, 0}})
	index.Add(Document{ID: "b", Text: "python", Embedding: []float64{0, 1}})
	results, err := index.Search([]float64{1, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Document.ID != "a" {
		t.Fatalf("results = %#v", results)
	}
}
