package evals

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	nego "github.com/gakon/nego-ai"
)

type Suite struct {
	Cases []Case `json:"cases"`
}

type Case struct {
	Name      string         `json:"name"`
	Prompt    string         `json:"prompt,omitempty"`
	Messages  []nego.Message `json:"messages,omitempty"`
	Exact     string         `json:"exact,omitempty"`
	Contains  string         `json:"contains,omitempty"`
	Regex     string         `json:"regex,omitempty"`
	JSONValid bool           `json:"json_valid,omitempty"`
	MaxTokens int            `json:"max_tokens,omitempty"`
}

type Report struct {
	Passed  int          `json:"passed"`
	Failed  int          `json:"failed"`
	Results []CaseResult `json:"results"`
}

type Summary struct {
	Passed      int           `json:"passed"`
	Failed      int           `json:"failed"`
	Total       int           `json:"total"`
	Duration    time.Duration `json:"duration"`
	PassRate    float64       `json:"pass_rate"`
	FailedCases []string      `json:"failed_cases,omitempty"`
}

type Comparison struct {
	Baseline     Summary     `json:"baseline"`
	Candidate    Summary     `json:"candidate"`
	Regressions  []CaseDelta `json:"regressions,omitempty"`
	Improvements []CaseDelta `json:"improvements,omitempty"`
	Changed      []CaseDelta `json:"changed,omitempty"`
}

type CaseDelta struct {
	Name            string `json:"name"`
	BaselinePassed  bool   `json:"baseline_passed"`
	CandidatePassed bool   `json:"candidate_passed"`
	BaselineError   string `json:"baseline_error,omitempty"`
	CandidateError  string `json:"candidate_error,omitempty"`
}

type CaseResult struct {
	Name     string        `json:"name"`
	Passed   bool          `json:"passed"`
	Output   string        `json:"output"`
	Error    string        `json:"error,omitempty"`
	Duration time.Duration `json:"duration"`
}

func Summarize(report Report) Summary {
	summary := Summary{Passed: report.Passed, Failed: report.Failed, Total: len(report.Results)}
	for _, result := range report.Results {
		summary.Duration += result.Duration
		if !result.Passed {
			summary.FailedCases = append(summary.FailedCases, result.Name)
		}
	}
	if summary.Total > 0 {
		summary.PassRate = float64(summary.Passed) / float64(summary.Total)
	}
	return summary
}

func Compare(baseline, candidate Report) Comparison {
	comparison := Comparison{
		Baseline:  Summarize(baseline),
		Candidate: Summarize(candidate),
	}
	baselineByName := make(map[string]CaseResult, len(baseline.Results))
	for _, result := range baseline.Results {
		baselineByName[result.Name] = result
	}
	for _, candidateResult := range candidate.Results {
		baselineResult, ok := baselineByName[candidateResult.Name]
		if !ok || baselineResult.Passed == candidateResult.Passed {
			continue
		}
		delta := CaseDelta{
			Name:            candidateResult.Name,
			BaselinePassed:  baselineResult.Passed,
			CandidatePassed: candidateResult.Passed,
			BaselineError:   baselineResult.Error,
			CandidateError:  candidateResult.Error,
		}
		comparison.Changed = append(comparison.Changed, delta)
		if baselineResult.Passed && !candidateResult.Passed {
			comparison.Regressions = append(comparison.Regressions, delta)
		}
		if !baselineResult.Passed && candidateResult.Passed {
			comparison.Improvements = append(comparison.Improvements, delta)
		}
	}
	return comparison
}

func Run(ctx context.Context, model nego.Model, suite Suite) Report {
	report := Report{Results: make([]CaseResult, 0, len(suite.Cases))}
	for _, tc := range suite.Cases {
		result := runCase(ctx, model, tc)
		if result.Passed {
			report.Passed++
		} else {
			report.Failed++
		}
		report.Results = append(report.Results, result)
	}
	return report
}

func runCase(ctx context.Context, model nego.Model, tc Case) CaseResult {
	start := time.Now()
	result := CaseResult{Name: tc.Name}
	output, err := generate(ctx, model, tc)
	result.Duration = time.Since(start)
	result.Output = output
	if err != nil {
		result.Error = err.Error()
		return result
	}
	if err := match(tc, output); err != nil {
		result.Error = err.Error()
		return result
	}
	result.Passed = true
	return result
}

func generate(ctx context.Context, model nego.Model, tc Case) (string, error) {
	if len(tc.Messages) > 0 {
		resp, err := model.Chat(ctx, nego.ChatRequest{Messages: tc.Messages, MaxTokens: tc.MaxTokens})
		if err != nil {
			return "", err
		}
		return resp.Message.Content, nil
	}
	out, err := model.Generate(ctx, nego.GenerateRequest{Prompt: tc.Prompt, MaxTokens: tc.MaxTokens})
	if err != nil {
		return "", err
	}
	return out.Text, nil
}

func match(tc Case, output string) error {
	if tc.Exact != "" && output != tc.Exact {
		return fmt.Errorf("expected exact output %q", tc.Exact)
	}
	if tc.Contains != "" && !strings.Contains(output, tc.Contains) {
		return fmt.Errorf("expected output to contain %q", tc.Contains)
	}
	if tc.Regex != "" {
		ok, err := regexp.MatchString(tc.Regex, output)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("expected output to match regex %q", tc.Regex)
		}
	}
	if tc.JSONValid {
		var value any
		if err := json.Unmarshal([]byte(output), &value); err != nil {
			return fmt.Errorf("expected valid JSON: %w", err)
		}
	}
	return nil
}
