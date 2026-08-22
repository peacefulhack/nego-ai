package main

import (
	"context"
	"fmt"
	"log"
	"os"

	nego "github.com/gakon/nego-ai"
	_ "github.com/gakon/nego-ai/backends/openai"
	"github.com/gakon/nego-ai/embeddings"
)

func main() {
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Backend:  "openai-compatible",
		Endpoint: os.Getenv("OPENAI_BASE_URL"),
		Model:    os.Getenv("OPENAI_EMBEDDING_MODEL"),
		APIKey:   os.Getenv("OPENAI_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	resp, err := nego.Embed(context.Background(), model, nego.EmbeddingRequest{
		Input: []string{"go routines", "concurrent functions"},
	})
	if err != nil {
		log.Fatal(err)
	}
	score, err := embeddings.Cosine(resp.Embeddings[0], resp.Embeddings[1])
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(score)
}
