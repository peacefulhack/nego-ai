package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/gakon/nego-ai/share"
)

func main() {
	card, err := os.ReadFile("examples/8.share-trained-model/model-card-template.md")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Before sharing a trained model, include these files:")
	fmt.Println("- model weights or adapter files")
	fmt.Println("- tokenizer files")
	fmt.Println("- config files")
	fmt.Println("- README.md model card")
	fmt.Println()
	fmt.Println(string(card))

	manifest, err := share.BuildManifest(share.ManifestOptions{
		Path:      "examples/8.share-trained-model",
		RepoID:    "username/qwen3-sft",
		BaseModel: "Qwen/Qwen3-0.6B",
	})
	if err != nil {
		log.Fatal(err)
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Example share manifest:")
	fmt.Println(string(data))
}
