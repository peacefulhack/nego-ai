package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

func ResolveCacheDir(override string) (string, error) {
	if override != "" {
		return filepath.Abs(override)
	}
	if value := os.Getenv("HF_HUB_CACHE"); value != "" {
		return filepath.Abs(value)
	}
	if value := os.Getenv("HF_HOME"); value != "" {
		return filepath.Abs(filepath.Join(value, "hub"))
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Abs(filepath.Join(home, ".cache", "huggingface", "hub"))
}

func SnapshotDir(cacheDir, repoType, repoID, commit string) string {
	return filepath.Join(RepoDir(cacheDir, repoType, repoID), "snapshots", commit)
}

func SnapshotFilePath(cacheDir, repoType, repoID, commit, filename string) (string, error) {
	return SafeJoin(SnapshotDir(cacheDir, repoType, repoID, commit), filename)
}

func BlobPath(cacheDir, repoType, repoID, key string) string {
	return filepath.Join(RepoDir(cacheDir, repoType, repoID), "blobs", key)
}

func RepoDir(cacheDir, repoType, repoID string) string {
	return filepath.Join(cacheDir, repoPrefix(repoType)+"--"+strings.ReplaceAll(repoID, "/", "--"))
}

func WriteRef(cacheDir, repoType, repoID, revision, commit string) error {
	if strings.Contains(revision, "/") {
		return nil
	}
	path := filepath.Join(RepoDir(cacheDir, repoType, repoID), "refs", revision)
	if err := EnsureParent(path); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(commit), 0o644)
}

func ReadRef(cacheDir, repoType, repoID, revision string) (string, error) {
	if strings.Contains(revision, "/") {
		return "", os.ErrNotExist
	}
	data, err := os.ReadFile(filepath.Join(RepoDir(cacheDir, repoType, repoID), "refs", revision))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func LocalFilePath(localDir, filename string) (string, error) {
	return SafeJoin(localDir, filename)
}

type CopyProgress func(done, total int64)

func MaterializeFile(src, localDir, filename string) (string, error) {
	return MaterializeFileWithProgress(context.Background(), src, localDir, filename, nil)
}

func MaterializeFileWithProgress(ctx context.Context, src, localDir, filename string, progress CopyProgress) (string, error) {
	dst, err := LocalFilePath(localDir, filename)
	if err != nil {
		return "", err
	}
	if err := CopyFileWithProgress(ctx, src, dst, progress); err != nil {
		return "", err
	}
	return dst, nil
}

func CopyFile(src, dst string) error {
	return CopyFileWithProgress(context.Background(), src, dst, nil)
}

func CopyFileWithProgress(ctx context.Context, src, dst string, progress CopyProgress) error {
	if err := EnsureParent(dst); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	total := int64(0)
	if info, err := in.Stat(); err == nil {
		total = info.Size()
	}

	tmp, err := os.CreateTemp(filepath.Dir(dst), ".nego-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	copyErr := copyWithProgress(ctx, tmp, in, total, progress)
	closeErr := tmp.Close()
	if copyErr != nil {
		_ = os.Remove(tmpPath)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmpPath)
		return closeErr
	}
	return ReplaceFile(tmpPath, dst)
}

func copyWithProgress(ctx context.Context, dst io.Writer, src io.Reader, total int64, progress CopyProgress) error {
	if progress != nil {
		progress(0, total)
	}
	buf := make([]byte, 1024*256)
	var done int64
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			done += int64(n)
			if progress != nil {
				progress(done, total)
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func ReplaceFile(tmpPath, dst string) error {
	if err := EnsureParent(dst); err != nil {
		return err
	}
	_ = os.Remove(dst)
	return os.Rename(tmpPath, dst)
}

func EnsureParent(filename string) error {
	return os.MkdirAll(filepath.Dir(filename), 0o755)
}

func FileExists(filename string) bool {
	info, err := os.Stat(filename)
	return err == nil && !info.IsDir()
}

func DirExists(dirname string) bool {
	info, err := os.Stat(dirname)
	return err == nil && info.IsDir()
}

func ValidateRepoPath(repoPath string) error {
	_, err := SafeJoin(string(filepath.Separator), repoPath)
	return err
}

func SafeJoin(base, repoPath string) (string, error) {
	if repoPath == "" {
		return "", fmt.Errorf("repo path cannot be empty")
	}
	clean := path.Clean(strings.ReplaceAll(repoPath, "\\", "/"))
	if clean == "." || strings.HasPrefix(clean, "../") || clean == ".." || strings.HasPrefix(clean, "/") {
		return "", fmt.Errorf("unsafe repo path %q", repoPath)
	}
	for _, part := range strings.Split(clean, "/") {
		if part == ".." || part == "" || strings.Contains(part, ":") {
			return "", fmt.Errorf("unsafe repo path %q", repoPath)
		}
	}
	return filepath.Join(base, filepath.FromSlash(clean)), nil
}

func BlobKey(etag, fallback string) string {
	key := strings.Trim(etag, "\"")
	key = strings.TrimPrefix(key, "W/")
	if key == "" {
		sum := sha256.Sum256([]byte(fallback))
		return hex.EncodeToString(sum[:])
	}
	for _, r := range key {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-' {
			continue
		}
		sum := sha256.Sum256([]byte(key))
		return hex.EncodeToString(sum[:])
	}
	return key
}

type Lock struct {
	path string
}

func AcquireLock(ctx context.Context, cacheDir, repoType, repoID, name string) (*Lock, error) {
	sum := sha256.Sum256([]byte(repoType + "/" + repoID + "/" + name))
	lockPath := filepath.Join(cacheDir, ".locks", hex.EncodeToString(sum[:])+".lock")
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o755); err != nil {
		return nil, err
	}
	for {
		f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			_, _ = fmt.Fprintf(f, "%d\n", os.Getpid())
			_ = f.Close()
			return &Lock{path: lockPath}, nil
		}
		if !os.IsExist(err) {
			return nil, err
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func (l *Lock) Release() {
	if l != nil && l.path != "" {
		_ = os.Remove(l.path)
	}
}

func repoPrefix(repoType string) string {
	switch repoType {
	case "dataset":
		return "datasets"
	case "space":
		return "spaces"
	default:
		return "models"
	}
}
