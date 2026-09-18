package adapters

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const maxLoRAFileBytes = 128 << 20

// LoadLinearLoRA reads a bounded, validated Nego linear adapter file. This is
// not a PEFT or GGUF adapter importer.
func LoadLinearLoRA(path string) (*LinearLoRA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() > maxLoRAFileBytes {
		return nil, fmt.Errorf("LoRA file exceeds %d bytes", maxLoRAFileBytes)
	}
	data, err := io.ReadAll(io.LimitReader(f, maxLoRAFileBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxLoRAFileBytes {
		return nil, fmt.Errorf("LoRA file exceeds %d bytes", maxLoRAFileBytes)
	}
	var l LinearLoRA
	if err := json.Unmarshal(data, &l); err != nil {
		return nil, err
	}
	if err := l.Validate(); err != nil {
		return nil, err
	}
	return &l, nil
}

// SaveLinearLoRA creates a new adapter file with private permissions. Existing
// files (including symlinks) are never overwritten; use a new checkpoint path.
func SaveLinearLoRA(path string, l *LinearLoRA) error {
	if path == "" {
		return fmt.Errorf("LoRA path is required")
	}
	if err := l.Validate(); err != nil {
		return err
	}
	data, err := json.Marshal(l)
	if err != nil {
		return err
	}
	if len(data) > maxLoRAFileBytes {
		return fmt.Errorf("LoRA file exceeds %d bytes", maxLoRAFileBytes)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(data)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}
