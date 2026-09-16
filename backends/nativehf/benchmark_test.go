package nativehf

import (
	"context"
	"testing"

	nego "github.com/gakon/nego-ai"
)

func BenchmarkForwardTokenTinyHF(b *testing.B) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: writeTinyForwardModel(b)})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := nativeModel.ForwardToken(0, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGenerateTinyHF(b *testing.B) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: writeTinyForwardModel(b)})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()

	req := nego.GenerateRequest{Prompt: "a", MaxTokens: 2}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Generate(context.Background(), req); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoadTensorFloat32CachedTinyHF(b *testing.B) {
	model, err := Backend{}.Load(context.Background(), nego.ModelOptions{Path: writeTinyForwardModel(b)})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	nativeModel := model.(*Model)
	if _, _, err := nativeModel.LoadTensorFloat32("model.norm.weight"); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := nativeModel.LoadTensorFloat32("model.norm.weight"); err != nil {
			b.Fatal(err)
		}
	}
}
