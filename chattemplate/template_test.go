package chattemplate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRenderChatML(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte("<|im_start|>{{ role }}"), 0o644); err != nil {
		t.Fatal(err)
	}
	prompt, err := Render(dir, []Message{
		{Role: RoleSystem, Content: "You are helpful."},
		{Role: RoleUser, Content: "Hello"},
	}, Options{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"<|im_start|>system\nYou are helpful.<|im_end|>", "<|im_start|>user\nHello<|im_end|>", "<|im_start|>assistant\n"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("expected %q in prompt:\n%s", want, prompt)
		}
	}
}

func TestLoadTemplateFromTokenizerConfig(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"chat_template":"<|im_start|>{{ role }}"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	template, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(template.Source, "<|im_start|>") {
		t.Fatalf("unexpected template: %#v", template)
	}
}

func TestRenderFallback(t *testing.T) {
	prompt, err := (&Template{}).Render([]Message{{Role: RoleUser, Content: "Hello"}}, Options{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	if prompt != "USER: Hello\nASSISTANT: " {
		t.Fatalf("prompt = %q", prompt)
	}
}
