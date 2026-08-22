package rag

import (
	"sort"
	"strings"

	"github.com/gakon/nego-ai/embeddings"
)

type Chunk struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type ChunkOptions struct {
	MaxRunes int
	Overlap  int
}

func ChunkText(text string, opts ChunkOptions) []Chunk {
	maxRunes := opts.MaxRunes
	if maxRunes <= 0 {
		maxRunes = 800
	}
	overlap := opts.Overlap
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= maxRunes {
		overlap = maxRunes / 4
	}
	runes := []rune(strings.TrimSpace(text))
	if len(runes) == 0 {
		return nil
	}
	var chunks []Chunk
	for start := 0; start < len(runes); {
		end := start + maxRunes
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, Chunk{ID: chunkID(len(chunks)), Text: string(runes[start:end])})
		if end == len(runes) {
			break
		}
		start = end - overlap
	}
	return chunks
}

type Document struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Embedding []float64 `json:"embedding"`
}

type SearchResult struct {
	Document Document `json:"document"`
	Score    float64  `json:"score"`
}

type Index struct {
	documents []Document
}

func NewIndex() *Index {
	return &Index{}
}

func (i *Index) Add(document Document) {
	i.documents = append(i.documents, document)
}

func (i *Index) Search(query []float64, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 5
	}
	results := make([]SearchResult, 0, len(i.documents))
	for _, document := range i.documents {
		score, err := embeddings.Cosine(query, document.Embedding)
		if err != nil {
			return nil, err
		}
		results = append(results, SearchResult{Document: document, Score: score})
	}
	sort.Slice(results, func(a, b int) bool {
		return results[a].Score > results[b].Score
	})
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func chunkID(index int) string {
	const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
	if index == 0 {
		return "chunk-0"
	}
	var buf [16]byte
	i := len(buf)
	n := index
	for n > 0 {
		i--
		buf[i] = alphabet[n%len(alphabet)]
		n /= len(alphabet)
	}
	return "chunk-" + string(buf[i:])
}
