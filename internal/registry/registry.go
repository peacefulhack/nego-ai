package registry

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/gakon/nego-ai/internal/cache"
)

type Entry struct {
	RepoID       string    `json:"repo_id"`
	RepoType     string    `json:"repo_type"`
	Revision     string    `json:"revision"`
	Commit       string    `json:"commit"`
	LocalDir     string    `json:"local_dir,omitempty"`
	SnapshotPath string    `json:"snapshot_path"`
	FileCount    int       `json:"file_count"`
	TotalSize    int64     `json:"total_size"`
	Files        []File    `json:"files,omitempty"`
	DownloadedAt time.Time `json:"downloaded_at"`
}

type File struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

type Store struct {
	cacheDir string
}

func NewStore(cacheDir string) *Store {
	return &Store{cacheDir: cacheDir}
}

func (s *Store) Upsert(entry Entry) error {
	if entry.RepoType == "" {
		entry.RepoType = "model"
	}
	if entry.DownloadedAt.IsZero() {
		entry.DownloadedAt = time.Now().UTC()
	}
	if entry.FileCount == 0 && len(entry.Files) > 0 {
		entry.FileCount = len(entry.Files)
	}
	if entry.TotalSize == 0 {
		for _, file := range entry.Files {
			entry.TotalSize += file.Size
		}
	}

	entries, err := s.List()
	if err != nil {
		return err
	}
	key := entryKey(entry.RepoType, entry.RepoID, entry.Revision)
	replaced := false
	for i := range entries {
		if entryKey(entries[i].RepoType, entries[i].RepoID, entries[i].Revision) == key {
			entries[i] = entry
			replaced = true
			break
		}
	}
	if !replaced {
		entries = append(entries, entry)
	}
	return s.write(entries)
}

func (s *Store) List() ([]Entry, error) {
	data, err := os.ReadFile(s.path())
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var entries []Entry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].RepoID == entries[j].RepoID {
			return entries[i].Revision < entries[j].Revision
		}
		return entries[i].RepoID < entries[j].RepoID
	})
	return entries, nil
}

func (s *Store) path() string {
	return filepath.Join(s.cacheDir, ".nego", "registry.json")
}

func (s *Store) write(entries []Entry) error {
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].RepoID == entries[j].RepoID {
			return entries[i].Revision < entries[j].Revision
		}
		return entries[i].RepoID < entries[j].RepoID
	})
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	path := s.path()
	if err := cache.EnsureParent(path); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".registry-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return cache.ReplaceFile(tmpPath, path)
}

func entryKey(repoType, repoID, revision string) string {
	return repoType + "\x00" + repoID + "\x00" + revision
}
