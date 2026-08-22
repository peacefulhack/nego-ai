package modelinfo

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type RuntimeFile struct {
	Path string `json:"path"`
	Kind string `json:"kind"`
	Size int64  `json:"size"`
}

func ResolveRuntimeFile(path string, kinds ...string) (string, error) {
	file, err := FindRuntimeFile(path, kinds...)
	if err != nil {
		return "", err
	}
	return file.Path, nil
}

func FindRuntimeFile(path string, kinds ...string) (*RuntimeFile, error) {
	if path == "" {
		return nil, fmt.Errorf("model path is required")
	}
	allowed := runtimeKindSet(kinds)
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	stat, err := os.Stat(absPath)
	if err != nil {
		return nil, err
	}
	if !stat.IsDir() {
		kind := fileKind(absPath)
		if !allowed[kind] {
			return nil, fmt.Errorf("model file %q is not a supported runtime file", absPath)
		}
		return &RuntimeFile{Path: absPath, Kind: kind, Size: stat.Size()}, nil
	}

	files, err := inspectFiles(absPath)
	if err != nil {
		return nil, err
	}
	candidates := make([]RuntimeFile, 0, len(files))
	for _, file := range files {
		if allowed[file.Kind] {
			candidates = append(candidates, RuntimeFile{
				Path: filepath.Join(absPath, filepath.FromSlash(file.Path)),
				Kind: file.Kind,
				Size: file.Size,
			})
		}
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no supported runtime file found under %q", absPath)
	}
	sortRuntimeFiles(candidates, kinds)
	return &candidates[0], nil
}

func runtimeKindSet(kinds []string) map[string]bool {
	if len(kinds) == 0 {
		kinds = []string{"gguf", "onnx"}
	}
	allowed := make(map[string]bool, len(kinds))
	for _, kind := range kinds {
		kind = strings.ToLower(strings.TrimSpace(kind))
		if kind != "" {
			allowed[kind] = true
		}
	}
	return allowed
}

func sortRuntimeFiles(files []RuntimeFile, kinds []string) {
	priority := make(map[string]int, len(kinds))
	for i, kind := range kinds {
		priority[strings.ToLower(kind)] = i
	}
	sort.Slice(files, func(i, j int) bool {
		left, right := files[i], files[j]
		if priority[left.Kind] != priority[right.Kind] {
			return priority[left.Kind] < priority[right.Kind]
		}
		leftDepth := strings.Count(filepath.ToSlash(left.Path), "/")
		rightDepth := strings.Count(filepath.ToSlash(right.Path), "/")
		if leftDepth != rightDepth {
			return leftDepth < rightDepth
		}
		if left.Size != right.Size {
			return left.Size > right.Size
		}
		return left.Path < right.Path
	})
}
