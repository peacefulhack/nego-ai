package main

import (
	"context"
	"fmt"
	"log"
	"os"

	nego "github.com/gakon/nego-ai"
	_ "github.com/gakon/nego-ai/backends/openai"
)

func main() {
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Backend:  "openai-compatible",
		Endpoint: os.Getenv("OPENAI_BASE_URL"),
		Model:    os.Getenv("OPENAI_MODEL"),
		APIKey:   os.Getenv("OPENAI_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	resp, err := model.Chat(context.Background(), nego.ChatRequest{
		Messages: []nego.Message{{Role: nego.RoleUser, Content: "Explain goroutines in one sentence."}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Message.Content)
}
