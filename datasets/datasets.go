package datasets

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

type Row map[string]any

type FilterOp string

const (
	FilterExists      FilterOp = "exists"
	FilterEqual       FilterOp = "eq"
	FilterNotEqual    FilterOp = "ne"
	FilterContains    FilterOp = "contains"
	FilterNotContains FilterOp = "not_contains"
)

type Filter struct {
	Field string
	Op    FilterOp
	Value string
}

type DedupeOptions struct {
	Keys       []string
	TrimSpace  bool
	IgnoreCase bool
}

type DedupeResult struct {
	Rows    []Row
	Removed int
}

type QualityOptions struct {
	Format     string
	Required   []string
	DedupeKeys []string
	TrimSpace  bool
	IgnoreCase bool
}

type QualityReport struct {
	Rows       int            `json:"rows"`
	Format     string         `json:"format"`
	Valid      bool           `json:"valid"`
	Error      string         `json:"error,omitempty"`
	Fields     []FieldQuality `json:"fields,omitempty"`
	Duplicates int            `json:"duplicates,omitempty"`
}

type FieldQuality struct {
	Field   string `json:"field"`
	Present int    `json:"present"`
	Missing int    `json:"missing"`
	Empty   int    `json:"empty"`
}

func ReadJSONL(r io.Reader) ([]Row, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var rows []Row
	line := 0
	for scanner.Scan() {
		line++
		text := scanner.Text()
		if text == "" {
			continue
		}
		var row Row
		if err := json.Unmarshal([]byte(text), &row); err != nil {
			return nil, fmt.Errorf("jsonl line %d: %w", line, err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func WriteJSONL(w io.Writer, rows []Row) error {
	encoder := json.NewEncoder(w)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

func ReadFile(path string) ([]Row, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	switch strings.ToLower(filepath.Ext(path)) {
	case ".csv":
		return ReadCSV(file)
	default:
		return ReadJSONL(file)
	}
}

func ReadCSV(r io.Reader) ([]Row, error) {
	reader := csv.NewReader(r)
	headers, err := reader.Read()
	if err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}
	var rows []Row
	rowNumber := 1
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		rowNumber++
		if len(record) != len(headers) {
			return nil, fmt.Errorf("csv row %d has %d fields, expected %d", rowNumber, len(record), len(headers))
		}
		row := make(Row, len(headers))
		for j, header := range headers {
			row[header] = record[j]
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func Split(rows []Row, testRatio float64, seed int64) ([]Row, []Row) {
	if testRatio < 0 {
		testRatio = 0
	}
	if testRatio > 1 {
		testRatio = 1
	}
	copied := append([]Row(nil), rows...)
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(copied), func(i, j int) {
		copied[i], copied[j] = copied[j], copied[i]
	})
	testCount := int(float64(len(copied)) * testRatio)
	test := append([]Row(nil), copied[:testCount]...)
	train := append([]Row(nil), copied[testCount:]...)
	return train, test
}

func Shuffle(rows []Row, seed int64) []Row {
	copied := append([]Row(nil), rows...)
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(copied), func(i, j int) {
		copied[i], copied[j] = copied[j], copied[i]
	})
	return copied
}

func SelectFields(rows []Row, fields []string) []Row {
	if len(fields) == 0 {
		return append([]Row(nil), rows...)
	}
	selected := make([]Row, 0, len(rows))
	for _, row := range rows {
		out := make(Row, len(fields))
		for _, field := range fields {
			if value, ok := row[field]; ok {
				out[field] = value
			}
		}
		selected = append(selected, out)
	}
	return selected
}

func RequireFields(rows []Row, fields []string) []Row {
	if len(fields) == 0 {
		return append([]Row(nil), rows...)
	}
	filtered := make([]Row, 0, len(rows))
	for _, row := range rows {
		if hasRequiredFields(row, fields) {
			filtered = append(filtered, row)
		}
	}
	return filtered
}

func ParseFilter(expr string) (Filter, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return Filter{}, fmt.Errorf("empty filter expression")
	}
	for _, op := range []struct {
		text string
		op   FilterOp
	}{
		{"!~", FilterNotContains},
		{"!=", FilterNotEqual},
		{"~", FilterContains},
		{"=", FilterEqual},
	} {
		if idx := strings.Index(expr, op.text); idx >= 0 {
			field := strings.TrimSpace(expr[:idx])
			value := strings.TrimSpace(expr[idx+len(op.text):])
			if field == "" {
				return Filter{}, fmt.Errorf("filter %q is missing a field", expr)
			}
			return Filter{Field: field, Op: op.op, Value: value}, nil
		}
	}
	return Filter{Field: expr, Op: FilterExists}, nil
}

func FilterRows(rows []Row, filters []Filter) []Row {
	if len(filters) == 0 {
		return append([]Row(nil), rows...)
	}
	filtered := make([]Row, 0, len(rows))
	for _, row := range rows {
		if matchFilters(row, filters) {
			filtered = append(filtered, row)
		}
	}
	return filtered
}

func DedupeRows(rows []Row, opts DedupeOptions) DedupeResult {
	seen := make(map[string]bool, len(rows))
	out := make([]Row, 0, len(rows))
	keys := cleanDedupeKeys(opts.Keys)
	for _, row := range rows {
		key := dedupeRowKey(row, keys, opts)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, row)
	}
	return DedupeResult{Rows: out, Removed: len(rows) - len(out)}
}

func AnalyzeQuality(rows []Row, opts QualityOptions) QualityReport {
	format := strings.ToLower(strings.TrimSpace(opts.Format))
	if format == "" {
		format = "auto"
	}
	required := cleanDedupeKeys(opts.Required)
	if len(required) == 0 {
		required = defaultRequiredFields(format)
	}
	report := QualityReport{
		Rows:   len(rows),
		Format: format,
		Valid:  true,
		Fields: analyzeFieldQuality(rows, required),
	}
	if len(opts.DedupeKeys) > 0 {
		report.Duplicates = DedupeRows(rows, DedupeOptions{
			Keys:       opts.DedupeKeys,
			TrimSpace:  opts.TrimSpace,
			IgnoreCase: opts.IgnoreCase,
		}).Removed
	}
	if err := ValidateFormat(rows, format); err != nil {
		report.Valid = false
		report.Error = err.Error()
		return report
	}
	for _, field := range report.Fields {
		if field.Missing > 0 || field.Empty > 0 {
			report.Valid = false
			report.Error = fmt.Sprintf("field %q has missing=%d empty=%d", field.Field, field.Missing, field.Empty)
			return report
		}
	}
	return report
}

func analyzeFieldQuality(rows []Row, fields []string) []FieldQuality {
	out := make([]FieldQuality, 0, len(fields))
	for _, field := range fields {
		quality := FieldQuality{Field: field}
		for _, row := range rows {
			value, ok := row[field]
			if !ok {
				quality.Missing++
				continue
			}
			quality.Present++
			if strings.TrimSpace(valueString(value)) == "" {
				quality.Empty++
			}
		}
		out = append(out, quality)
	}
	return out
}

func defaultRequiredFields(format string) []string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "completion":
		return []string{"prompt", "completion"}
	case "instruction":
		return []string{"instruction", "output"}
	case "chat":
		return []string{"messages"}
	default:
		return nil
	}
}

func cleanDedupeKeys(keys []string) []string {
	out := make([]string, 0, len(keys))
	seen := make(map[string]bool, len(keys))
	for _, key := range keys {
		key = strings.TrimSpace(key)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, key)
	}
	return out
}

func dedupeRowKey(row Row, keys []string, opts DedupeOptions) string {
	var value any
	if len(keys) == 0 {
		value = normalizeDedupeValue(row, opts)
	} else {
		values := make([]any, len(keys))
		for i, key := range keys {
			values[i] = normalizeDedupeValue(row[key], opts)
		}
		value = values
	}
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(data)
}

func normalizeDedupeValue(value any, opts DedupeOptions) any {
	switch typed := value.(type) {
	case string:
		if opts.TrimSpace {
			typed = strings.TrimSpace(typed)
		}
		if opts.IgnoreCase {
			typed = strings.ToLower(typed)
		}
		return typed
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = normalizeDedupeValue(item, opts)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			out[key] = normalizeDedupeValue(item, opts)
		}
		return out
	default:
		return typed
	}
}

func ValidateRequiredFields(rows []Row, fields ...string) error {
	for i, row := range rows {
		for _, field := range fields {
			if _, ok := row[field]; !ok {
				return fmt.Errorf("row %d missing required field %q", i, field)
			}
		}
	}
	return nil
}

func hasRequiredFields(row Row, fields []string) bool {
	for _, field := range fields {
		value, ok := row[field]
		if !ok || strings.TrimSpace(valueString(value)) == "" {
			return false
		}
	}
	return true
}

func matchFilters(row Row, filters []Filter) bool {
	for _, filter := range filters {
		value, ok := row[filter.Field]
		switch filter.Op {
		case FilterExists:
			if !ok || strings.TrimSpace(valueString(value)) == "" {
				return false
			}
		case FilterEqual:
			if !ok || valueString(value) != filter.Value {
				return false
			}
		case FilterNotEqual:
			if ok && valueString(value) == filter.Value {
				return false
			}
		case FilterContains:
			if !ok || !strings.Contains(valueString(value), filter.Value) {
				return false
			}
		case FilterNotContains:
			if ok && strings.Contains(valueString(value), filter.Value) {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func valueString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	case fmt.Stringer:
		return typed.String()
	default:
		return fmt.Sprint(typed)
	}
}

func ValidateFormat(rows []Row, format string) error {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "auto":
		for i, row := range rows {
			if err := validateAutoRow(row); err != nil {
				return fmt.Errorf("row %d: %w", i, err)
			}
		}
		return nil
	case "chat":
		for i, row := range rows {
			if err := validateChatRow(row); err != nil {
				return fmt.Errorf("row %d: %w", i, err)
			}
		}
		return nil
	case "completion":
		return ValidateRequiredFields(rows, "prompt", "completion")
	case "instruction":
		return ValidateRequiredFields(rows, "instruction", "output")
	default:
		return fmt.Errorf("unsupported dataset format %q", format)
	}
}

func validateAutoRow(row Row) error {
	if _, ok := row["messages"]; ok {
		return validateChatRow(row)
	}
	if _, ok := row["prompt"]; ok {
		if _, ok := row["completion"]; ok {
			return nil
		}
		if _, ok := row["response"]; ok {
			return nil
		}
		return fmt.Errorf("completion row requires completion or response")
	}
	if _, ok := row["instruction"]; ok {
		if _, ok := row["output"]; ok {
			return nil
		}
		return fmt.Errorf("instruction row requires output")
	}
	return fmt.Errorf("unknown row format")
}

func validateChatRow(row Row) error {
	raw, ok := row["messages"]
	if !ok {
		return fmt.Errorf("missing required field %q", "messages")
	}
	messages, ok := raw.([]any)
	if !ok || len(messages) == 0 {
		return fmt.Errorf("messages must be a non-empty array")
	}
	for i, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok {
			return fmt.Errorf("messages[%d] must be an object", i)
		}
		role, _ := message["role"].(string)
		content, _ := message["content"].(string)
		if role != "system" && role != "user" && role != "assistant" {
			return fmt.Errorf("messages[%d] has unsupported role %q", i, role)
		}
		if content == "" {
			return fmt.Errorf("messages[%d] content is required", i)
		}
	}
	return nil
}
