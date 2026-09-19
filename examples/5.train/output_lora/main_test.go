package main

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/gakon/nego-ai/datasets"
)

func testSource(sequences, forwards *int) featureSource {
	return featureSource{
		encode: func(text string) ([]int, error) {
			ids := make([]int, len(text))
			for i := range text {
				ids[i] = int(text[i])
			}
			return ids, nil
		},
		newSequence: func() (featureStep, error) {
			*sequences++
			return func(id int) ([]float32, []float32, error) {
				*forwards++
				return []float32{float32(id)}, make([]float32, 256), nil
			}, nil
		},
	}
}

func TestCollectMasksPromptAndResetsHistory(t *testing.T) {
	sequences, forwards := 0, 0
	source := testSource(&sequences, &forwards)
	samples, err := collect(context.Background(), source, []datasets.Row{{"prompt": "p", "completion": "ab"}, {"prompt": "q", "completion": "c"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 3 || sequences != 2 || forwards != 5 {
		t.Fatalf("samples=%d sequences=%d forwards=%d", len(samples), sequences, forwards)
	}
	for i, want := range []int{'a', 'b', 'c'} {
		if samples[i].Target != want {
			t.Fatalf("target %d = %d, want %d", i, samples[i].Target, want)
		}
	}
	if samples[0].Input[0] != '\n' || samples[1].Input[0] != 'a' || samples[2].Input[0] != '\n' {
		t.Fatal("features must precede their target token")
	}
}

func TestCollectRejectsInvalidDataAndCancellation(t *testing.T) {
	for _, row := range []datasets.Row{{}, {"prompt": 1, "completion": "a"}, {"prompt": "p", "completion": ""}, {"prompt": "p", "completion": strings.Repeat("x", 128)}} {
		sequences, forwards := 0, 0
		if _, err := collect(context.Background(), testSource(&sequences, &forwards), []datasets.Row{row}); err == nil {
			t.Fatalf("accepted bad row %v", row)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	sequences, forwards := 0, 0
	if _, err := collect(ctx, testSource(&sequences, &forwards), []datasets.Row{{"prompt": "p", "completion": "a"}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if forwards != 0 {
		t.Fatal("forward ran after cancellation")
	}
}

func TestCollectRejectsUnstableTokenBoundary(t *testing.T) {
	sequences, forwards := 0, 0
	source := testSource(&sequences, &forwards)
	source.encode = func(text string) ([]int, error) {
		if strings.HasSuffix(text, "\n") {
			return []int{1}, nil
		}
		return []int{2, 3}, nil
	}
	if _, err := collect(context.Background(), source, []datasets.Row{{"prompt": "p", "completion": "a"}}); err == nil {
		t.Fatal("accepted ambiguous prompt/completion token boundary")
	}
}
