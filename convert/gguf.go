package convert

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
)

type GGUFOptions struct {
	Converter string
	ModelDir  string
	Output    string
	Quantize  string
}

func ConvertGGUF(ctx context.Context, opts GGUFOptions) error {
	if opts.Converter == "" {
		return fmt.Errorf("converter is required")
	}
	if opts.ModelDir == "" {
		return fmt.Errorf("model directory is required")
	}
	if opts.Output == "" {
		return fmt.Errorf("output path is required")
	}
	args := []string{opts.ModelDir, "--outfile", opts.Output}
	if opts.Quantize != "" {
		args = append(args, "--outtype", opts.Quantize)
	}
	cmd := exec.CommandContext(ctx, opts.Converter, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if stderr.Len() > 0 {
			return fmt.Errorf("gguf conversion failed: %s", stderr.String())
		}
		return fmt.Errorf("gguf conversion failed: %w", err)
	}
	return nil
}
