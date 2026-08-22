package chattemplate

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

type Options struct {
	AddGenerationPrompt bool
}

type Template struct {
	Path   string
	Source string
}

func Load(path string) (*Template, error) {
	root, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if info, err := os.Stat(root); err == nil && !info.IsDir() {
		data, err := os.ReadFile(root)
		if err != nil {
			return nil, err
		}
		return &Template{Path: root, Source: string(data)}, nil
	}
	if data, err := os.ReadFile(filepath.Join(root, "chat_template.jinja")); err == nil {
		return &Template{Path: filepath.Join(root, "chat_template.jinja"), Source: string(data)}, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	source, err := chatTemplateFromTokenizerConfig(filepath.Join(root, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}
	return &Template{Path: filepath.Join(root, "tokenizer_config.json"), Source: source}, nil
}

func (t *Template) Render(messages []Message, opts Options) (string, error) {
	for _, message := range messages {
		if message.Role != RoleSystem && message.Role != RoleUser && message.Role != RoleAssistant {
			return "", fmt.Errorf("unsupported chat role %q", message.Role)
		}
	}
	switch {
	case strings.Contains(t.Source, "<|im_start|>"):
		return renderChatML(messages, opts), nil
	case strings.Contains(t.Source, "[INST]"):
		return renderLlamaInst(messages, opts), nil
	default:
		return renderFallback(messages, opts), nil
	}
}

func Render(path string, messages []Message, opts Options) (string, error) {
	template, err := Load(path)
	if err != nil {
		return "", err
	}
	return template.Render(messages, opts)
}

func chatTemplateFromTokenizerConfig(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	var body struct {
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(data, &body); err != nil {
		return "", err
	}
	return body.ChatTemplate, nil
}

func renderChatML(messages []Message, opts Options) string {
	var b strings.Builder
	for _, message := range messages {
		fmt.Fprintf(&b, "<|im_start|>%s\n%s<|im_end|>\n", message.Role, message.Content)
	}
	if opts.AddGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
	}
	return b.String()
}

func renderLlamaInst(messages []Message, opts Options) string {
	var system string
	var b strings.Builder
	for _, message := range messages {
		switch message.Role {
		case RoleSystem:
			system = message.Content
		case RoleUser:
			b.WriteString("[INST] ")
			if system != "" {
				b.WriteString("<<SYS>>\n")
				b.WriteString(system)
				b.WriteString("\n<</SYS>>\n\n")
				system = ""
			}
			b.WriteString(message.Content)
			b.WriteString(" [/INST]")
		case RoleAssistant:
			b.WriteString(" ")
			b.WriteString(message.Content)
			b.WriteString(" ")
		}
	}
	if opts.AddGenerationPrompt && !strings.HasSuffix(b.String(), " ") {
		b.WriteString(" ")
	}
	return b.String()
}

func renderFallback(messages []Message, opts Options) string {
	var b strings.Builder
	for _, message := range messages {
		fmt.Fprintf(&b, "%s: %s\n", strings.ToUpper(string(message.Role)), message.Content)
	}
	if opts.AddGenerationPrompt {
		b.WriteString("ASSISTANT: ")
	}
	return b.String()
}
