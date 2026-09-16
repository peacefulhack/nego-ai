package tokenizer

import (
	"fmt"
	"sort"
	"strings"
)

type FixtureProfile struct {
	Name        string      `json:"profile"`
	Description string      `json:"description,omitempty"`
	Cases       []CheckCase `json:"cases"`
}

var curatedFixtureProfiles = map[string]FixtureProfile{
	"basic": {
		Name:        "basic",
		Description: "Small tokenizer sanity checks for plain text, whitespace, and code-shaped prompts.",
		Cases: []CheckCase{
			{Name: "plain english", Text: "Hello world"},
			{Name: "newline", Text: "Hello\nworld"},
			{Name: "code indentation", Text: "func main() {\n\treturn\n}"},
		},
	},
	"llama": {
		Name:        "llama",
		Description: "Llama-style prompt and text cases for local tokenizer regression checks.",
		Cases: []CheckCase{
			{Name: "plain english", Text: "Hello world"},
			{Name: "number and punctuation", Text: "The answer is 42."},
			{Name: "llama chat markers", Text: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"},
		},
	},
	"qwen": {
		Name:        "qwen",
		Description: "Qwen-style multilingual and chat-marker cases for local tokenizer regression checks.",
		Cases: []CheckCase{
			{Name: "plain english", Text: "Hello world"},
			{Name: "multilingual", Text: "Halo dunia. 你好，世界。"},
			{Name: "qwen chat markers", Text: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"},
		},
	},
}

func FixtureProfileNames() []string {
	names := make([]string, 0, len(curatedFixtureProfiles))
	for name := range curatedFixtureProfiles {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func CuratedCheckCases(profile string) ([]CheckCase, error) {
	fixture, err := CuratedFixtureProfile(profile)
	if err != nil {
		return nil, err
	}
	return fixture.Cases, nil
}

func CuratedFixtureProfile(profile string) (FixtureProfile, error) {
	key := strings.ToLower(strings.TrimSpace(profile))
	fixture, ok := curatedFixtureProfiles[key]
	if !ok {
		return FixtureProfile{}, fmt.Errorf("unknown tokenizer fixture profile %q; available profiles: %s", profile, strings.Join(FixtureProfileNames(), ", "))
	}
	return cloneFixtureProfile(fixture), nil
}

func BuildFixtureProfile(profile string, tok TextCodec) (FixtureProfile, error) {
	fixture, err := CuratedFixtureProfile(profile)
	if err != nil {
		return FixtureProfile{}, err
	}
	for i := range fixture.Cases {
		if tok == nil {
			text := fixture.Cases[i].Text
			fixture.Cases[i].Decoded = &text
			continue
		}
		ids, err := tok.Encode(fixture.Cases[i].Text)
		if err != nil {
			return FixtureProfile{}, fmt.Errorf("encode fixture case %q: %w", fixture.Cases[i].Name, err)
		}
		decoded, err := tok.Decode(ids)
		if err != nil {
			return FixtureProfile{}, fmt.Errorf("decode fixture case %q: %w", fixture.Cases[i].Name, err)
		}
		fixture.Cases[i].Tokens = append([]int(nil), ids...)
		fixture.Cases[i].IDs = nil
		fixture.Cases[i].Decoded = &decoded
	}
	return fixture, nil
}

func cloneFixtureProfile(fixture FixtureProfile) FixtureProfile {
	out := FixtureProfile{
		Name:        fixture.Name,
		Description: fixture.Description,
		Cases:       make([]CheckCase, len(fixture.Cases)),
	}
	copy(out.Cases, fixture.Cases)
	for i := range out.Cases {
		out.Cases[i].Tokens = append([]int(nil), out.Cases[i].Tokens...)
		out.Cases[i].IDs = append([]int(nil), out.Cases[i].IDs...)
		if out.Cases[i].Decoded != nil {
			decoded := *out.Cases[i].Decoded
			out.Cases[i].Decoded = &decoded
		}
	}
	return out
}
