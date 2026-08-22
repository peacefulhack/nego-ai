package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gakon/nego-ai/hub"
)

func main() {
	path, err := hub.DownloadFile(context.Background(), hub.DownloadFileOptions{
		RepoID:   "Qwen/Qwen3-0.6B",
		Filename: "tokenizer.json",
		LocalDir: "./models/qwen3",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)
}
