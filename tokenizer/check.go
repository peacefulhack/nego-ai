package tokenizer

import "fmt"

type TextCodec interface {
	Encode(string) ([]int, error)
	Decode([]int) (string, error)
}

type CheckCase struct {
	Name    string  `json:"name,omitempty"`
	Text    string  `json:"text"`
	Tokens  []int   `json:"tokens,omitempty"`
	IDs     []int   `json:"ids,omitempty"`
	Decoded *string `json:"decoded,omitempty"`
}

type CheckCaseResult struct {
	Name            string   `json:"name,omitempty"`
	Text            string   `json:"text"`
	Tokens          []int    `json:"tokens,omitempty"`
	ExpectedTokens  []int    `json:"expected_tokens,omitempty"`
	Decoded         string   `json:"decoded,omitempty"`
	ExpectedDecoded *string  `json:"expected_decoded,omitempty"`
	Passed          bool     `json:"passed"`
	Errors          []string `json:"errors,omitempty"`
}

type CheckResult struct {
	Passed int               `json:"passed"`
	Failed int               `json:"failed"`
	Total  int               `json:"total"`
	Cases  []CheckCaseResult `json:"cases"`
}

func ValidateCheckCases(cases []CheckCase) ([]CheckCase, error) {
	if len(cases) == 0 {
		return nil, fmt.Errorf("tokenize check fixture has no cases")
	}
	for i, c := range cases {
		if len(c.Tokens) == 0 && len(c.IDs) == 0 && c.Decoded == nil {
			return nil, fmt.Errorf("tokenize check case %d must set tokens, ids, or decoded", i+1)
		}
	}
	return cases, nil
}

func CheckCases(tok TextCodec, cases []CheckCase) CheckResult {
	result := CheckResult{Total: len(cases), Cases: make([]CheckCaseResult, 0, len(cases))}
	for i, c := range cases {
		name := c.Name
		if name == "" {
			name = fmt.Sprintf("case %d", i+1)
		}
		caseResult := CheckCaseResult{
			Name:            name,
			Text:            c.Text,
			ExpectedTokens:  expectedTokenIDs(c),
			ExpectedDecoded: c.Decoded,
			Passed:          true,
		}
		ids, err := tok.Encode(c.Text)
		if err != nil {
			caseResult.Passed = false
			caseResult.Errors = append(caseResult.Errors, "encode: "+err.Error())
		} else {
			caseResult.Tokens = ids
			if len(caseResult.ExpectedTokens) > 0 && !equalInts(ids, caseResult.ExpectedTokens) {
				caseResult.Passed = false
				caseResult.Errors = append(caseResult.Errors, fmt.Sprintf("tokens got %v want %v", ids, caseResult.ExpectedTokens))
			}
			decoded, err := tok.Decode(ids)
			if err != nil {
				caseResult.Passed = false
				caseResult.Errors = append(caseResult.Errors, "decode: "+err.Error())
			} else {
				caseResult.Decoded = decoded
				if c.Decoded != nil && decoded != *c.Decoded {
					caseResult.Passed = false
					caseResult.Errors = append(caseResult.Errors, fmt.Sprintf("decoded got %q want %q", decoded, *c.Decoded))
				}
			}
		}
		if caseResult.Passed {
			result.Passed++
		} else {
			result.Failed++
		}
		result.Cases = append(result.Cases, caseResult)
	}
	return result
}

func expectedTokenIDs(c CheckCase) []int {
	if len(c.Tokens) > 0 {
		return c.Tokens
	}
	return c.IDs
}

func equalInts(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
