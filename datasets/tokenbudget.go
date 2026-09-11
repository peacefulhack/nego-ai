package datasets

import (
	"fmt"
	"sort"
	"strings"

	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/tokenizer"
)

type TokenBudgetOptions struct {
	ModelPath  string
	Format     string
	MaxContext int
}

type TokenBudgetReport struct {
	Format        string           `json:"format"`
	Rows          int              `json:"rows"`
	CountedRows   int              `json:"counted_rows"`
	Valid         bool             `json:"valid"`
	MaxContext    int              `json:"max_context,omitempty"`
	TotalTokens   int              `json:"total_tokens"`
	MinTokens     int              `json:"min_tokens"`
	MaxTokens     int              `json:"max_tokens"`
	AverageTokens float64          `json:"average_tokens"`
	OverLimit     int              `json:"over_limit"`
	RowsDetail    []TokenBudgetRow `json:"rows_detail"`
}

type TokenBudgetRow struct {
	Row       int    `json:"row"`
	Format    string `json:"format"`
	Tokens    int    `json:"tokens,omitempty"`
	OverLimit bool   `json:"over_limit,omitempty"`
	Error     string `json:"error,omitempty"`
}

func AnalyzeTokenBudget(rows []Row, opts TokenBudgetOptions) (TokenBudgetReport, error) {
	format := normalizeFormat(opts.Format)
	if format == "" {
		format = "auto"
	}
	if !supportedFormat(format) {
		return TokenBudgetReport{}, fmt.Errorf("unsupported dataset format %q", opts.Format)
	}
	if strings.TrimSpace(opts.ModelPath) == "" {
		return TokenBudgetReport{}, fmt.Errorf("model path is required")
	}
	tok, err := tokenizer.Load(opts.ModelPath)
	if err != nil {
		return TokenBudgetReport{}, err
	}
	template, err := chattemplate.Load(opts.ModelPath)
	if err != nil {
		return TokenBudgetReport{}, err
	}
	report := TokenBudgetReport{
		Format:     format,
		Rows:       len(rows),
		Valid:      true,
		MaxContext: opts.MaxContext,
		MinTokens:  -1,
		RowsDetail: make([]TokenBudgetRow, 0, len(rows)),
	}
	for i, row := range rows {
		detail := TokenBudgetRow{Row: i + 1}
		text, rowFormat, err := datasetTrainingText(row, format, template)
		detail.Format = rowFormat
		if err != nil {
			detail.Error = err.Error()
			report.Valid = false
			report.RowsDetail = append(report.RowsDetail, detail)
			continue
		}
		count, err := tok.Count(text)
		if err != nil {
			detail.Error = err.Error()
			report.Valid = false
			report.RowsDetail = append(report.RowsDetail, detail)
			continue
		}
		detail.Tokens = count
		report.CountedRows++
		if opts.MaxContext > 0 && count > opts.MaxContext {
			detail.OverLimit = true
			report.OverLimit++
			report.Valid = false
		}
		report.TotalTokens += count
		if report.MinTokens < 0 || count < report.MinTokens {
			report.MinTokens = count
		}
		if count > report.MaxTokens {
			report.MaxTokens = count
		}
		report.RowsDetail = append(report.RowsDetail, detail)
	}
	if report.MinTokens < 0 {
		report.MinTokens = 0
	}
	if report.CountedRows > 0 {
		report.AverageTokens = float64(report.TotalTokens) / float64(report.CountedRows)
	}
	return report, nil
}

func LongestTokenRows(rows []TokenBudgetRow, n int) []TokenBudgetRow {
	if n <= 0 {
		return nil
	}
	out := append([]TokenBudgetRow(nil), rows...)
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Tokens > out[j].Tokens
	})
	if len(out) > n {
		out = out[:n]
	}
	return out
}

func datasetTrainingText(row Row, format string, template *chattemplate.Template) (string, string, error) {
	if format == "auto" {
		format = inferRowFormat(row)
	}
	switch format {
	case "chat":
		messages, err := rowMessages(row)
		if err != nil {
			return "", format, err
		}
		prompt, err := template.Render(messages, chattemplate.Options{})
		if err != nil {
			return "", format, err
		}
		return prompt, format, nil
	case "completion":
		prompt := strings.TrimSpace(valueString(row["prompt"]))
		completion := strings.TrimSpace(valueString(row["completion"]))
		if completion == "" {
			completion = strings.TrimSpace(valueString(row["response"]))
		}
		if prompt == "" || completion == "" {
			return "", format, fmt.Errorf("completion row requires prompt and completion or response")
		}
		return prompt + "\n" + completion, format, nil
	case "instruction":
		instruction := strings.TrimSpace(valueString(row["instruction"]))
		output := strings.TrimSpace(valueString(row["output"]))
		if instruction == "" || output == "" {
			return "", format, fmt.Errorf("instruction row requires instruction and output")
		}
		input := strings.TrimSpace(valueString(row["input"]))
		if input != "" {
			return "Instruction: " + instruction + "\nInput: " + input + "\nOutput: " + output, format, nil
		}
		return "Instruction: " + instruction + "\nOutput: " + output, format, nil
	default:
		return "", format, fmt.Errorf("unknown row format")
	}
}

func rowMessages(row Row) ([]chattemplate.Message, error) {
	raw, ok := row["messages"]
	if !ok {
		return nil, fmt.Errorf("missing required field %q", "messages")
	}
	values, ok := raw.([]any)
	if !ok || len(values) == 0 {
		return nil, fmt.Errorf("messages must be a non-empty array")
	}
	messages := make([]chattemplate.Message, 0, len(values))
	for i, value := range values {
		rawMessage, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("messages[%d] must be an object", i)
		}
		role := chattemplate.Role(valueString(rawMessage["role"]))
		content := valueString(rawMessage["content"])
		if role != chattemplate.RoleSystem && role != chattemplate.RoleUser && role != chattemplate.RoleAssistant {
			return nil, fmt.Errorf("messages[%d] has unsupported role %q", i, role)
		}
		if strings.TrimSpace(content) == "" {
			return nil, fmt.Errorf("messages[%d] content is required", i)
		}
		messages = append(messages, chattemplate.Message{Role: role, Content: content})
	}
	return messages, nil
}

func inferRowFormat(row Row) string {
	switch {
	case row["messages"] != nil:
		return "chat"
	case row["prompt"] != nil && (row["completion"] != nil || row["response"] != nil):
		return "completion"
	case row["instruction"] != nil && row["output"] != nil:
		return "instruction"
	default:
		return "unknown"
	}
}

func normalizeFormat(format string) string {
	return strings.ToLower(strings.TrimSpace(format))
}

func supportedFormat(format string) bool {
	switch format {
	case "auto", "chat", "completion", "instruction":
		return true
	default:
		return false
	}
}
