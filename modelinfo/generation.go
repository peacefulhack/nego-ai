package modelinfo

import (
	"encoding/json"
	"math"
	"strconv"
)

type GenerationConfig struct {
	MaxLength         *int           `json:"max_length,omitempty"`
	MaxNewTokens      *int           `json:"max_new_tokens,omitempty"`
	MinNewTokens      *int           `json:"min_new_tokens,omitempty"`
	DoSample          *bool          `json:"do_sample,omitempty"`
	Temperature       *float64       `json:"temperature,omitempty"`
	TopP              *float64       `json:"top_p,omitempty"`
	TopK              *int           `json:"top_k,omitempty"`
	TypicalP          *float64       `json:"typical_p,omitempty"`
	RepetitionPenalty *float64       `json:"repetition_penalty,omitempty"`
	NoRepeatNGramSize *int           `json:"no_repeat_ngram_size,omitempty"`
	BOSTokenID        *int           `json:"bos_token_id,omitempty"`
	PadTokenID        *int           `json:"pad_token_id,omitempty"`
	EOSTokenIDs       []int          `json:"eos_token_ids,omitempty"`
	StopStrings       []string       `json:"stop_strings,omitempty"`
	Extra             map[string]any `json:"extra,omitempty"`
}

func ParseGenerationConfig(raw map[string]any) *GenerationConfig {
	if len(raw) == 0 {
		return nil
	}
	cfg := &GenerationConfig{}
	known := map[string]bool{
		"max_length":           true,
		"max_new_tokens":       true,
		"min_new_tokens":       true,
		"do_sample":            true,
		"temperature":          true,
		"top_p":                true,
		"top_k":                true,
		"typical_p":            true,
		"repetition_penalty":   true,
		"no_repeat_ngram_size": true,
		"bos_token_id":         true,
		"pad_token_id":         true,
		"eos_token_id":         true,
		"stop_strings":         true,
	}
	cfg.MaxLength = intPtrFromAny(raw["max_length"])
	cfg.MaxNewTokens = intPtrFromAny(raw["max_new_tokens"])
	cfg.MinNewTokens = intPtrFromAny(raw["min_new_tokens"])
	cfg.DoSample = boolPtrFromAny(raw["do_sample"])
	cfg.Temperature = floatPtrFromAny(raw["temperature"])
	cfg.TopP = floatPtrFromAny(raw["top_p"])
	cfg.TopK = intPtrFromAny(raw["top_k"])
	cfg.TypicalP = floatPtrFromAny(raw["typical_p"])
	cfg.RepetitionPenalty = floatPtrFromAny(raw["repetition_penalty"])
	cfg.NoRepeatNGramSize = intPtrFromAny(raw["no_repeat_ngram_size"])
	cfg.BOSTokenID = intPtrFromAny(raw["bos_token_id"])
	cfg.PadTokenID = intPtrFromAny(raw["pad_token_id"])
	cfg.EOSTokenIDs = intSliceFromAny(raw["eos_token_id"])
	cfg.StopStrings = stringSliceFromAnyFlexible(raw["stop_strings"])
	for key, value := range raw {
		if known[key] || !isJSONSerializable(value) {
			continue
		}
		if cfg.Extra == nil {
			cfg.Extra = make(map[string]any)
		}
		cfg.Extra[key] = value
	}
	if cfg.empty() {
		return nil
	}
	return cfg
}

func (c *GenerationConfig) empty() bool {
	return c == nil ||
		(c.MaxLength == nil &&
			c.MaxNewTokens == nil &&
			c.MinNewTokens == nil &&
			c.DoSample == nil &&
			c.Temperature == nil &&
			c.TopP == nil &&
			c.TopK == nil &&
			c.TypicalP == nil &&
			c.RepetitionPenalty == nil &&
			c.NoRepeatNGramSize == nil &&
			c.BOSTokenID == nil &&
			c.PadTokenID == nil &&
			len(c.EOSTokenIDs) == 0 &&
			len(c.StopStrings) == 0 &&
			len(c.Extra) == 0)
}

func intPtrFromAny(value any) *int {
	switch v := value.(type) {
	case nil:
		return nil
	case int:
		return &v
	case int64:
		if v < math.MinInt || v > math.MaxInt {
			return nil
		}
		out := int(v)
		return &out
	case float64:
		if v != math.Trunc(v) || v < math.MinInt || v > math.MaxInt {
			return nil
		}
		out := int(v)
		return &out
	case json.Number:
		i, err := strconv.ParseInt(v.String(), 10, 0)
		if err != nil {
			return nil
		}
		out := int(i)
		return &out
	case string:
		i, err := strconv.Atoi(v)
		if err != nil {
			return nil
		}
		return &i
	default:
		return nil
	}
}

func floatPtrFromAny(value any) *float64 {
	switch v := value.(type) {
	case nil:
		return nil
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil
		}
		return &v
	case json.Number:
		f, err := strconv.ParseFloat(v.String(), 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return nil
		}
		return &f
	case string:
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return nil
		}
		return &f
	default:
		if i := intPtrFromAny(value); i != nil {
			f := float64(*i)
			return &f
		}
		return nil
	}
}

func boolPtrFromAny(value any) *bool {
	switch v := value.(type) {
	case nil:
		return nil
	case bool:
		return &v
	case string:
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil
		}
		return &b
	default:
		return nil
	}
}

func intSliceFromAny(value any) []int {
	if one := intPtrFromAny(value); one != nil {
		return []int{*one}
	}
	values, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]int, 0, len(values))
	for _, value := range values {
		if i := intPtrFromAny(value); i != nil {
			out = append(out, *i)
		}
	}
	return out
}

func stringSliceFromAnyFlexible(value any) []string {
	switch v := value.(type) {
	case nil:
		return nil
	case string:
		if v == "" {
			return nil
		}
		return []string{v}
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return v
	default:
		return nil
	}
}

func isJSONSerializable(value any) bool {
	switch v := value.(type) {
	case nil, bool, string, float64, json.Number:
		return true
	case []any:
		for _, item := range v {
			if !isJSONSerializable(item) {
				return false
			}
		}
		return true
	case map[string]any:
		for _, item := range v {
			if !isJSONSerializable(item) {
				return false
			}
		}
		return true
	default:
		return false
	}
}
