package main

import (
	"log"
	"os"

	"github.com/gakon/nego-ai/datasets"
)

func main() {
	rows, err := datasets.ReadFile("examples/3.prepare-dataset/completion-data.csv")
	if err != nil {
		log.Fatal(err)
	}

	rows = datasets.RequireFields(rows, []string{"prompt", "completion"})
	rows = datasets.SelectFields(rows, []string{"prompt", "completion"})

	if err := datasets.WriteJSONL(os.Stdout, rows); err != nil {
		log.Fatal(err)
	}
}
