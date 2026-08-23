package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gakon/nego-ai/hub"
)

func main() {
	ctx := context.Background()
	path, err := hub.DownloadFile(ctx, hub.DownloadFileOptions{
		RepoID:   "Qwen/Qwen3-0.6B",
		Filename: "tokenizer.json",
		LocalDir: "./models/qwen3",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)

	gguf, err := hub.ResolveGGUFFile(ctx, hub.ResolveGGUFFileOptions{
		RepoID: "Qwen/Qwen3-0.6B",
		Quant:  "Q4_K_M",
	})
	if err != nil {
		log.Fatal(err)
	}
	path, err = hub.DownloadFile(ctx, hub.DownloadFileOptions{
		RepoID:   gguf.RepoID,
		Filename: gguf.Filename,
		LocalDir: "./models/qwen3-gguf",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)
}
