package training

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type NativeManifestEntry struct {
	Path         string         `json:"path"`
	ManifestPath string         `json:"manifest_path"`
	ModifiedAt   time.Time      `json:"modified_at"`
	CreatedAt    time.Time      `json:"created_at,omitempty"`
	Manifest     NativeManifest `json:"manifest"`
}

type NativeManifestDiscoveryOptions struct {
	Root      string
	BaseModel string
}

func ListNativeManifests(opts NativeManifestDiscoveryOptions) ([]NativeManifestEntry, error) {
	root := strings.TrimSpace(opts.Root)
	if root == "" {
		root = "./outputs"
	}
	stat, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var entries []NativeManifestEntry
	if !stat.IsDir() {
		entry, ok, err := loadNativeManifestEntry(root, opts.BaseModel)
		if err != nil {
			return nil, err
		}
		if ok {
			entries = append(entries, entry)
		}
		sortNativeManifestEntries(entries)
		return entries, nil
	}
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if filepath.Base(path) != "manifest.json" {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil
		}
		entry, ok, err := loadNativeManifestEntry(path, opts.BaseModel)
		if err != nil {
			return err
		}
		if ok {
			entries = append(entries, entry)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sortNativeManifestEntries(entries)
	return entries, nil
}

func LatestNativeManifest(opts NativeManifestDiscoveryOptions) (NativeManifestEntry, error) {
	entries, err := ListNativeManifests(opts)
	if err != nil {
		return NativeManifestEntry{}, err
	}
	if len(entries) == 0 {
		root := strings.TrimSpace(opts.Root)
		if root == "" {
			root = "./outputs"
		}
		if strings.TrimSpace(opts.BaseModel) != "" {
			return NativeManifestEntry{}, fmt.Errorf("no native training manifests found under %s for base model %s", root, opts.BaseModel)
		}
		return NativeManifestEntry{}, fmt.Errorf("no native training manifests found under %s", root)
	}
	return entries[0], nil
}

func loadNativeManifestEntry(path, baseModel string) (NativeManifestEntry, bool, error) {
	manifestPath, err := NativeManifestPath(path)
	if err != nil {
		if os.IsNotExist(err) {
			return NativeManifestEntry{}, false, nil
		}
		return NativeManifestEntry{}, false, err
	}
	stat, err := os.Lstat(manifestPath)
	if err != nil {
		return NativeManifestEntry{}, false, err
	}
	if stat.Mode()&os.ModeSymlink != 0 || stat.Size() > maxNativeManifestBytes {
		return NativeManifestEntry{}, false, nil
	}
	manifest, err := LoadNativeManifest(manifestPath)
	if err != nil {
		return NativeManifestEntry{}, false, nil
	}
	if wanted := strings.TrimSpace(baseModel); wanted != "" && filepath.Clean(manifest.BaseModel) != filepath.Clean(wanted) {
		return NativeManifestEntry{}, false, nil
	}
	return NativeManifestEntry{
		Path:         filepath.Dir(manifestPath),
		ManifestPath: manifestPath,
		ModifiedAt:   stat.ModTime(),
		CreatedAt:    manifest.CreatedAt,
		Manifest:     manifest,
	}, true, nil
}

func sortNativeManifestEntries(entries []NativeManifestEntry) {
	sort.Slice(entries, func(i, j int) bool {
		left := nativeManifestSortTime(entries[i])
		right := nativeManifestSortTime(entries[j])
		if !left.Equal(right) {
			return left.After(right)
		}
		return entries[i].ManifestPath < entries[j].ManifestPath
	})
}

func nativeManifestSortTime(entry NativeManifestEntry) time.Time {
	if !entry.CreatedAt.IsZero() {
		return entry.CreatedAt
	}
	return entry.ModifiedAt
}
