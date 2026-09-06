package main

import (
	"context"
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
	result, err := training.RunNative(context.Background(), training.NativeOptions{
		BaseModel:     baseModel,
		TrainFile:     "examples/5.train/train.jsonl",
		EvalFile:      "examples/5.train/test.jsonl",
		DatasetFormat: "completion",
		OutputDir:     "./outputs/qwen3-token-bias",
		Method:        "token-bias",
		Epochs:        1,
		LearningRate:  0.1,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Native adapter written")
	fmt.Printf("Base model: %s\n", result.BaseModel)
	fmt.Printf("Train file: %s (%d rows)\n", result.TrainFile, result.TrainRows)
	fmt.Printf("Adapter: %s\n", result.AdapterPath)
}
