package main

import (
	"context"
	"log"
	"net/http"
	"os"

	nego "github.com/gakon/nego-ai"
	_ "github.com/gakon/nego-ai/backends/llama"
	_ "github.com/gakon/nego-ai/backends/native"
	_ "github.com/gakon/nego-ai/backends/nativehf"
	"github.com/gakon/nego-ai/server"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run ./examples/6.serve <model-path>")
	}
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Path: os.Args[1],
	})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	handler, err := server.NewHandler(server.HandlerOptions{ModelID: "local-model", Model: model})
	if err != nil {
		log.Fatal(err)
	}
	log.Fatal(http.ListenAndServe(":8080", handler))
}
