package main

import (
	"bytes"
	"context"
	_ "embed"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/adapters"
	"github.com/gakon/nego-ai/backends/native"
	"github.com/gakon/nego-ai/datasets"
	"github.com/gakon/nego-ai/modelinfo"
	"github.com/gakon/nego-ai/training"
)

//go:embed train.jsonl
var exampleData []byte

func main() {
	modelPath := flag.String("model", "./models/qwen3-gguf", "local GGUF base model")
	output := flag.String("out", "./outputs/qwen3-output-lora.json", "new output checkpoint path")
	epochs := flag.Int("epochs", 3, "SGD epochs over frozen features")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	if err := run(ctx, *modelPath, *output, *epochs); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, path, output string, epochs int) error {
	if epochs <= 0 {
		return fmt.Errorf("epochs must be positive")
	}
	if _, err := os.Lstat(output); err == nil {
		return fmt.Errorf("checkpoint already exists; choose a new -out path")
	} else if !os.IsNotExist(err) {
		return err
	}
	fmt.Println("1. Loading the downloaded GGUF model on CPU...")
	loaded, err := nego.LoadModel(ctx, nego.ModelOptions{Backend: native.BackendName, Path: path})
	if err != nil {
		return err
	}
	defer loaded.Close()
	m := loaded.(*native.Model)
	rows, err := datasets.ReadJSONL(bytes.NewReader(exampleData))
	if err != nil {
		return err
	}
	fmt.Println("2. Collecting frozen features from train.jsonl...")
	samples, err := collect(ctx, m, rows)
	if err != nil {
		return err
	}
	if len(samples) == 0 {
		return fmt.Errorf("dataset contains no completion targets")
	}
	lora, err := adapters.NewLinearLoRA(len(samples[0].Input), len(samples[0].BaseLogits), 4, 4, 42)
	if err != nil {
		return err
	}
	fmt.Printf("3. Training output-head LoRA on %d target tokens...\n", len(samples))
	result, err := training.FitLinearLoRA(ctx, lora, samples, training.LinearLoRAOptions{
		Epochs: epochs, LearningRate: .01, MaxGradNorm: 1,
		Progress: func(p training.LinearLoRAProgress) {
			fmt.Printf("   Epoch %d/%d: training loss %.4f\n", p.Epoch, p.Epochs, p.MeanLoss)
		},
	})
	if err != nil {
		return err
	}
	if err := adapters.SaveLinearLoRA(output, lora); err != nil {
		return err
	}
	fmt.Printf("4. Saved %s; training loss %.4f -> %.4f (not held-out quality).\n", output, result.InitialLoss, result.FinalLoss)
	if err := loaded.Close(); err != nil {
		return err
	}
	fmt.Println("5. Reloading the same base GGUF with the saved output adapter...")
	adapted, err := nego.LoadModel(ctx, nego.ModelOptions{Backend: native.BackendName, Path: path, Options: map[string]string{"output_lora": output}})
	if err != nil {
		return err
	}
	defer adapted.Close()
	reply, err := adapted.Chat(ctx, nego.ChatRequest{Messages: []nego.Message{{Role: nego.RoleUser, Content: "What is Go? /no_think"}}, MaxTokens: 32})
	if err != nil {
		return err
	}
	fmt.Println(reply.Message.Content)
	fmt.Println("Only the output adapter was trained. Base weights are unchanged; this is not full-transformer LoRA.")
	return nil
}

func collect(ctx context.Context, m *native.Model, rows []datasets.Row) ([]training.LinearLoRASample, error) {
	var samples []training.LinearLoRASample
	for row, data := range rows {
		prompt, ok := data["prompt"].(string)
		if !ok || prompt == "" {
			return nil, fmt.Errorf("row %d needs a non-empty prompt string", row+1)
		}
		completion, ok := data["completion"].(string)
		if !ok || completion == "" {
			return nil, fmt.Errorf("row %d needs a non-empty completion string", row+1)
		}
		options := modelinfo.EncodeOptions{AddBOS: m.Vocab().AddBOS}
		prefix, err := m.Vocab().Encode(prompt+"\n", options)
		if err != nil {
			return nil, err
		}
		ids, err := m.Vocab().Encode(prompt+"\n"+completion, options)
		if err != nil {
			return nil, err
		}
		// Reject ambiguous tokenizer boundaries rather than training prompt labels.
		if len(prefix) == 0 || len(ids) <= len(prefix) {
			return nil, fmt.Errorf("row %d has no completion tokens", row+1)
		}
		for i := range prefix {
			if ids[i] != prefix[i] {
				return nil, fmt.Errorf("row %d has an unstable prompt/completion token boundary", row+1)
			}
		}
		if len(ids) > 128 || len(samples)+len(ids)-len(prefix) > 128 {
			return nil, fmt.Errorf("example is limited to 128 tokens per row and 128 completion targets")
		}
		state, err := native.NewDecodeState(m.Spec())
		if err != nil {
			return nil, err
		}
		for i := 0; i+1 < len(ids); i++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			hidden, logits, err := m.ForwardFeaturesWithState(ids[i], state)
			if err != nil {
				return nil, err
			}
			if i+1 >= len(prefix) {
				samples = append(samples, training.LinearLoRASample{Input: hidden, BaseLogits: logits, Target: ids[i+1]})
			}
		}
		fmt.Printf("   Row %d/%d: %d completion targets\n", row+1, len(rows), len(ids)-len(prefix))
	}
	return samples, nil
}
