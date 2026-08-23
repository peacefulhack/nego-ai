package convert

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

type GGUFOptions struct {
	Converter string
	Python    string
	ModelDir  string
	Output    string
	Quantize  string
}

func ConvertGGUF(ctx context.Context, opts GGUFOptions) error {
	opts.Converter = resolveConverter(opts.Converter)
	if opts.Converter == "" {
		return fmt.Errorf("Nego does not natively convert Hugging Face safetensors to GGUF yet; install llama.cpp and pass --converter /path/to/convert_hf_to_gguf.py or set NEGO_LLAMA_CONVERTER")
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
	command := opts.Converter
	if opts.Python != "" {
		args = append([]string{opts.Converter}, args...)
		command = opts.Python
	}
	if dir := filepath.Dir(opts.Output); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	cmd := exec.CommandContext(ctx, command, args...)
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

func resolveConverter(value string) string {
	if value != "" {
		return value
	}
	if value := os.Getenv("NEGO_LLAMA_CONVERTER"); value != "" {
		return value
	}
	return os.Getenv("NEGO_LLAMA_CONVERT")
}
