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
