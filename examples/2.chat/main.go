package main

import (
	"context"
	"fmt"
	"log"
	"os"

	nego "github.com/gakon/nego-ai"
	_ "github.com/gakon/nego-ai/backends/native"
	_ "github.com/gakon/nego-ai/backends/nativehf"
)

func main() {
	path := "./models/qwen3"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Path: path,
	})
	if err != nil {
		log.Fatalf("load local model %s: %v", path, err)
	}
	defer model.Close()

	resp, err := model.Chat(context.Background(), nego.ChatRequest{
		Messages:  []nego.Message{{Role: nego.RoleUser, Content: "Explain goroutines in one sentence."}},
		MaxTokens: 16,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Message.Content)
}
