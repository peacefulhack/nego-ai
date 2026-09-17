package cli

import (
	"context"
	"fmt"
	"io"

	nego "github.com/gakon/nego-ai"
)

type statusRuntimeSmoke struct {
	Passed  bool   `json:"passed"`
	Backend string `json:"backend,omitempty"`
	Prompt  string `json:"prompt"`
	Output  string `json:"output,omitempty"`
	Error   string `json:"error,omitempty"`
}

func runStatusRuntimeSmoke(ctx context.Context, path, prompt string) (result statusRuntimeSmoke, stats *nego.RuntimeStats) {
	result.Prompt = prompt
	if err := ctx.Err(); err != nil {
		result.Error = err.Error()
		return
	}
	model, err := nego.LoadModel(ctx, nego.ModelOptions{
		Backend: nego.BackendNativeAuto,
		Path:    path,
	})
	if err != nil {
		result.Error = fmt.Sprintf("load model: %v", err)
		return
	}
	defer func() {
		if err := model.Close(); err != nil && result.Error == "" {
			result.Passed = false
			result.Error = fmt.Sprintf("close model: %v", err)
		}
	}()
	output, err := model.Generate(ctx, nego.GenerateRequest{Prompt: prompt, MaxTokens: 1})
	if loadedStats, ok := nego.RuntimeStatsOf(model); ok {
		stats = &loadedStats
		result.Backend = loadedStats.Backend
	}
	if err != nil {
		result.Error = fmt.Sprintf("generate: %v", err)
		return
	}
	result.Passed = true
	result.Output = output.Text
	return
}

func writeRuntimeSmoke(w io.Writer, result statusRuntimeSmoke) {
	fmt.Fprintln(w, "Runtime smoke:")
	fmt.Fprintf(w, "  Passed:  %s\n", yesNo(result.Passed))
	if result.Backend != "" {
		fmt.Fprintf(w, "  Backend: %s\n", result.Backend)
	}
	fmt.Fprintf(w, "  Prompt:  %q\n", result.Prompt)
	if result.Passed {
		fmt.Fprintf(w, "  Output:  %q\n", result.Output)
	} else {
		fmt.Fprintf(w, "  Error:   %s\n", result.Error)
	}
}
