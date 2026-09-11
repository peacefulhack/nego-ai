package modelinfo

import "testing"

func TestParseGenerationConfig(t *testing.T) {
	cfg := ParseGenerationConfig(map[string]any{
		"max_new_tokens":       float64(128),
		"do_sample":            true,
		"temperature":          0.7,
		"top_p":                0.8,
		"top_k":                float64(20),
		"repetition_penalty":   "1.05",
		"bos_token_id":         float64(151643),
		"pad_token_id":         float64(151643),
		"eos_token_id":         []any{float64(151645), float64(151643)},
		"stop_strings":         []any{"<|im_end|>"},
		"transformers_version": "4.51.0",
	})
	if cfg == nil {
		t.Fatal("expected generation config")
	}
	if cfg.MaxNewTokens == nil || *cfg.MaxNewTokens != 128 {
		t.Fatalf("MaxNewTokens = %#v", cfg.MaxNewTokens)
	}
	if cfg.DoSample == nil || !*cfg.DoSample {
		t.Fatalf("DoSample = %#v", cfg.DoSample)
	}
	if cfg.Temperature == nil || *cfg.Temperature != 0.7 {
		t.Fatalf("Temperature = %#v", cfg.Temperature)
	}
	if cfg.TopP == nil || *cfg.TopP != 0.8 {
		t.Fatalf("TopP = %#v", cfg.TopP)
	}
	if cfg.TopK == nil || *cfg.TopK != 20 {
		t.Fatalf("TopK = %#v", cfg.TopK)
	}
	if cfg.RepetitionPenalty == nil || *cfg.RepetitionPenalty != 1.05 {
		t.Fatalf("RepetitionPenalty = %#v", cfg.RepetitionPenalty)
	}
	if len(cfg.EOSTokenIDs) != 2 || cfg.EOSTokenIDs[0] != 151645 || cfg.EOSTokenIDs[1] != 151643 {
		t.Fatalf("EOSTokenIDs = %#v", cfg.EOSTokenIDs)
	}
	if len(cfg.StopStrings) != 1 || cfg.StopStrings[0] != "<|im_end|>" {
		t.Fatalf("StopStrings = %#v", cfg.StopStrings)
	}
	if cfg.Extra["transformers_version"] != "4.51.0" {
		t.Fatalf("Extra = %#v", cfg.Extra)
	}
}

func TestParseGenerationConfigAllowsScalarEOS(t *testing.T) {
	cfg := ParseGenerationConfig(map[string]any{"eos_token_id": float64(2)})
	if cfg == nil || len(cfg.EOSTokenIDs) != 1 || cfg.EOSTokenIDs[0] != 2 {
		t.Fatalf("unexpected cfg: %#v", cfg)
	}
}
