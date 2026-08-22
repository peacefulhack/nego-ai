package datasets

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
)

type Row map[string]any

func ReadJSONL(r io.Reader) ([]Row, error) {
	scanner := bufio.NewScanner(r)
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

func ReadCSV(r io.Reader) ([]Row, error) {
	reader := csv.NewReader(r)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, nil
	}
	headers := records[0]
	rows := make([]Row, 0, len(records)-1)
	for i, record := range records[1:] {
		if len(record) != len(headers) {
			return nil, fmt.Errorf("csv row %d has %d fields, expected %d", i+2, len(record), len(headers))
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
