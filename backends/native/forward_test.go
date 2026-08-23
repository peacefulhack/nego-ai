package native

import (
	"strings"
	"testing"

	"github.com/gakon/nego-ai/modelinfo"
)

func TestPlanGenerationEncodesPromptWhenVocabExists(t *testing.T) {
	model := &Model{
		vocab: &modelinfo.GGUFVocab{Tokens: []string{"hello", "▁world"}},
	}
	plan, err := model.planGeneration("hello world", GenerationOptions{MaxTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Options.MaxTokens != 4 {
		t.Fatalf("unexpected options: %#v", plan.Options)
	}
	if len(plan.PromptTokenIDs) != 2 || plan.PromptTokenIDs[0] != 0 || plan.PromptTokenIDs[1] != 1 {
		t.Fatalf("unexpected token ids: %#v", plan.PromptTokenIDs)
	}
}

func TestPlanGenerationAllowsMissingVocabTokens(t *testing.T) {
	model := &Model{vocab: &modelinfo.GGUFVocab{}}
	plan, err := model.planGeneration("hello", GenerationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.PromptTokenIDs) != 0 {
		t.Fatalf("expected no prompt tokens, got %#v", plan.PromptTokenIDs)
	}
}

func TestPlanGenerationRejectsEmptyPrompt(t *testing.T) {
	model := &Model{}
	_, err := model.planGeneration(" \n\t", GenerationOptions{})
	if err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("unexpected error: %v", err)
	}
}
