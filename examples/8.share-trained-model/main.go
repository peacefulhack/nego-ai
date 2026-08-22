package main

import (
	"fmt"
	"log"
	"os"
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
}
