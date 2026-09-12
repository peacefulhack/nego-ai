package modelinfo

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const maxNativeAdapterManifestBytes = 1 << 20

type NativeAdapterInfo struct {
	Version            int               `json:"version"`
	Type               string            `json:"type"`
	BaseModel          string            `json:"base_model"`
	AdapterPath        string            `json:"adapter_path"`
	Method             string            `json:"method,omitempty"`
	DatasetFormat      string            `json:"dataset_format,omitempty"`
	RecommendedBackend string            `json:"recommended_backend,omitempty"`
	RuntimeOptions     map[string]string `json:"runtime_options,omitempty"`
	RunArgs            []string          `json:"run_args,omitempty"`
	ChatArgs           []string          `json:"chat_args,omitempty"`
	VocabSize          int               `json:"vocab_size,omitempty"`
	UpdatedTokens      int               `json:"updated_tokens,omitempty"`
	TrainTokens        int               `json:"train_tokens,omitempty"`
	CreatedAt          time.Time         `json:"created_at,omitempty"`
	ManifestPath       string            `json:"manifest_path,omitempty"`
}

func InspectNativeAdapter(path string) (*NativeAdapterInfo, error) {
	manifestPath, err := nativeAdapterManifestPath(path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	if len(data) > maxNativeAdapterManifestBytes {
		return nil, fmt.Errorf("native adapter manifest exceeds %d bytes", maxNativeAdapterManifestBytes)
	}
	var info NativeAdapterInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return nil, err
	}
	if info.Type != "nego-native-adapter" {
		return nil, nil
	}
	info.ManifestPath = manifestPath
	return &info, nil
}

func nativeAdapterManifestPath(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("native adapter path is required")
	}
	root, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	stat, err := os.Stat(root)
	if err != nil {
		return "", err
	}
	if stat.IsDir() {
		return filepath.Join(root, "manifest.json"), nil
	}
	if filepath.Base(root) == "manifest.json" {
		return root, nil
	}
	return "", os.ErrNotExist
}
