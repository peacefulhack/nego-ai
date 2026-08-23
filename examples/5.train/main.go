package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gakon/nego-ai/training"
)

func main() {
	baseModel := "./models/qwen3"
	if len(os.Args) > 1 {
		baseModel = os.Args[1]
	}
	if _, err := os.Stat(baseModel); err != nil {
		log.Fatalf("base model %s is missing; run step 1 first: go run ./cmd/nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3", baseModel)
	}
	job, err := training.NewLoRAJob(training.InitOptions{
		Name:      "qwen3-lora",
		BaseModel: baseModel,
		TrainFile: "examples/5.train/train.jsonl",
		EvalFile:  "examples/5.train/test.jsonl",
		OutputDir: "./outputs/qwen3-lora",
		WorkDir:   ".",
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := training.WriteJob("examples/5.train/train-job.json", job); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Training job written: examples/5.train/train-job.json")
	fmt.Printf("Base model: %s\n", job.BaseModel)
	fmt.Printf("Train file: %s\n", job.TrainFile)
	fmt.Printf("Output dir: %s\n", job.OutputDir)
}
