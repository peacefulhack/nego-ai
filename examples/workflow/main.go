package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/datasets"
	"github.com/gakon/nego-ai/evals"
	"github.com/gakon/nego-ai/training"
)

type chatSession struct {
	Backend  string         `json:"backend"`
	Path     string         `json:"path"`
	Messages []nego.Message `json:"messages"`
}

func main() {
	root := exampleRoot()

	printStep(1, "Download a base model", "nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3")
	printStep(2, "Inspect the downloaded model", "nego inspect ./models/qwen3")

	rows, err := datasets.ReadFile(filepath.Join(root, "completion-data.csv"))
	must(err)
	fmt.Printf("3. Prepare a dataset: loaded %d rows from completion-data.csv\n", len(rows))

	train, test := datasets.Split(rows, 0.34, 42)
	fmt.Printf("4. Validate and split the dataset: %d train rows, %d test rows\n", len(train), len(test))

	printStep(5, "Check tokenizer and context budget", `nego context ./models/qwen3 "Explain Go." --max-context 4096`)
	printStep(6, "Render a chat prompt", `nego prompt ./models/qwen3 --system "You are helpful." --user "Explain Go."`)
	printStep(7, "Chat with the model", "nego chat ./models/model.gguf --interactive --session chats/qwen.json")

	var suite evals.Suite
	readJSON(filepath.Join(root, "eval-suite.json"), &suite)
	fmt.Printf("8. Run a baseline eval: loaded %d eval cases from eval-suite.json\n", len(suite.Cases))

	var job training.JobSpec
	readJSON(filepath.Join(root, "train-job.json"), &job)
	fmt.Printf("9. Train or fine-tune: job %q runs %s with %d args\n", job.Name, job.Command, len(job.Args))

	printStep(10, "Inspect and evaluate the trained output", "nego inspect ./outputs/qwen3-sft")
	printStep(11, "Convert or optimize", "nego convert gguf ./outputs/qwen3-sft --out ./outputs/qwen3-sft.gguf --converter /path/to/convert_hf_to_gguf.py")

	var session chatSession
	readJSON(filepath.Join(root, "chat-session.json"), &session)
	fmt.Printf("12. Serve and share: sample session uses backend %q with %d messages\n", session.Backend, len(session.Messages))
}

func printStep(number int, title, command string) {
	fmt.Printf("%d. %s: %s\n", number, title, command)
}

func exampleRoot() string {
	candidates := []string{
		".",
		filepath.FromSlash("examples/workflow"),
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(filepath.Join(candidate, "completion-data.csv")); err == nil {
			return candidate
		}
	}
	return filepath.FromSlash("examples/workflow")
}

func readJSON(path string, out any) {
	data, err := os.ReadFile(path)
	must(err)
	must(json.Unmarshal(data, out))
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
