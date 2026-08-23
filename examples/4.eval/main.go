package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/evals"
)

func main() {
	data, err := os.ReadFile("examples/4.eval/eval-suite.json")
	if err != nil {
		log.Fatal(err)
	}
	var suite evals.Suite
	if err := json.Unmarshal(data, &suite); err != nil {
		log.Fatal(err)
	}

	report := evals.Run(context.Background(), exampleModel{}, suite)
	fmt.Printf("passed=%d failed=%d\n", report.Passed, report.Failed)
}

type exampleModel struct{}

func (exampleModel) Generate(_ context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	if strings.Contains(req.Prompt, "JSON") {
		return &nego.GenerateOutput{Text: `{"ok":true}`}, nil
	}
	return &nego.GenerateOutput{Text: "hello from Go"}, nil
}

func (exampleModel) Chat(context.Context, nego.ChatRequest) (*nego.ChatResponse, error) {
	return &nego.ChatResponse{Message: nego.Message{Role: nego.RoleAssistant, Content: "Nego uses Go."}}, nil
}

func (exampleModel) StreamChat(context.Context, nego.ChatRequest) (nego.Stream, error) {
	return nil, fmt.Errorf("streaming is not implemented in this example")
}

func (exampleModel) Close() error {
	return nil
}
