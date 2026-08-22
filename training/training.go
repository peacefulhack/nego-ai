package training

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"
)

type JobSpec struct {
	Name    string            `json:"name"`
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	WorkDir string            `json:"work_dir,omitempty"`
}

type Result struct {
	Name     string        `json:"name"`
	Success  bool          `json:"success"`
	Stdout   string        `json:"stdout"`
	Stderr   string        `json:"stderr"`
	Duration time.Duration `json:"duration"`
}

func Run(ctx context.Context, spec JobSpec) (Result, error) {
	if spec.Command == "" {
		return Result{Name: spec.Name}, fmt.Errorf("training command is required")
	}
	start := time.Now()
	cmd := exec.CommandContext(ctx, spec.Command, spec.Args...)
	cmd.Dir = spec.WorkDir
	cmd.Env = os.Environ()
	for key, value := range spec.Env {
		cmd.Env = append(cmd.Env, key+"="+value)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	result := Result{
		Name:     spec.Name,
		Success:  err == nil,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: time.Since(start),
	}
	if err != nil {
		return result, err
	}
	return result, nil
}
