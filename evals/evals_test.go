package evals

import (
	"context"
	"testing"

	nego "github.com/gakon/nego-ai"
)

func TestRunPassesMatchers(t *testing.T) {
	report := Run(context.Background(), evalModel{}, Suite{Cases: []Case{
		{Name: "contains", Prompt: "hello", Contains: "hello"},
		{Name: "regex", Prompt: "hello", Regex: "generated: .*"},
		{Name: "json", Prompt: "json", JSONValid: true},
	}})
	if report.Passed != 3 || report.Failed != 0 {
		t.Fatalf("report = %#v", report)
	}
}

func TestRunRecordsFailures(t *testing.T) {
	report := Run(context.Background(), evalModel{}, Suite{Cases: []Case{{Name: "fail", Prompt: "hello", Exact: "nope"}}})
	if report.Passed != 0 || report.Failed != 1 || report.Results[0].Error == "" {
		t.Fatalf("report = %#v", report)
	}
}

type evalModel struct{}

func (evalModel) Generate(_ context.Context, req nego.GenerateRequest) (*nego.GenerateOutput, error) {
	if req.Prompt == "json" {
		return &nego.GenerateOutput{Text: `{"ok":true}`}, nil
	}
	return &nego.GenerateOutput{Text: "generated: " + req.Prompt}, nil
}

func (evalModel) Chat(context.Context, nego.ChatRequest) (*nego.ChatResponse, error) {
	return &nego.ChatResponse{Message: nego.Message{Role: nego.RoleAssistant, Content: "chat"}}, nil
}

func (evalModel) StreamChat(context.Context, nego.ChatRequest) (nego.Stream, error) {
	return nil, nil
}

func (evalModel) Close() error {
	return nil
}
