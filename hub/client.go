package hub

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/gakon/nego-ai/internal/cache"
	"github.com/gakon/nego-ai/internal/hfhub"
	"github.com/gakon/nego-ai/internal/patterns"
)

func (c *Client) DownloadFile(ctx context.Context, opts DownloadFileOptions) (string, error) {
	if err := validateFileOptions(&opts); err != nil {
		return "", err
	}
	cfg, err := c.config(opts.CacheDir, opts.Token, opts.Progress)
	if err != nil {
		return "", err
	}
	api := c.api(cfg.token)
	repoType := string(normalizeRepoType(opts.RepoType))
	revision := normalizeRevision(opts.Revision)

	c.report(cfg.progress, ProgressEvent{RepoID: opts.RepoID, Filename: opts.Filename, State: "resolving"})

	if opts.LocalFilesOnly {
		return c.localFileOnly(opts, cfg.cacheDir, repoType, revision)
	}

	meta, err := api.HeadFile(ctx, repoType, opts.RepoID, revision, opts.Filename)
	if err != nil {
		return "", wrapHFError(err)
	}
	c.report(cfg.progress, ProgressEvent{
		RepoID:     opts.RepoID,
		Filename:   opts.Filename,
		BytesTotal: meta.Size,
		FilesTotal: 1,
		State:      "planned",
	})
	commit := firstNonEmpty(meta.Commit, revision)
	snapshotPath, err := cache.SnapshotFilePath(cfg.cacheDir, repoType, opts.RepoID, commit, opts.Filename)
	if err != nil {
		return "", err
	}
	if !opts.Force && cache.FileExists(snapshotPath) {
		return c.finishCachedFile(ctx, opts, snapshotPath, cfg.progress, meta)
	}

	path, err := c.downloadOne(ctx, api, cfg.cacheDir, repoType, opts.RepoID, commit, commit, opts.Filename, *meta, opts.Force, cfg.progress)
	if err != nil {
		return "", err
	}
	if opts.LocalDir != "" {
		path, err = c.materializeFile(ctx, opts.RepoID, opts.Filename, path, opts.LocalDir, cfg.progress)
		if err != nil {
			return "", err
		}
	}
	c.report(cfg.progress, ProgressEvent{RepoID: opts.RepoID, Filename: opts.Filename, State: "complete", Destination: path})
	return path, nil
}

func (c *Client) DownloadSnapshot(ctx context.Context, opts DownloadSnapshotOptions) (string, error) {
	if err := validateSnapshotOptions(&opts); err != nil {
		return "", err
	}
	cfg, err := c.config(opts.CacheDir, opts.Token, opts.Progress)
	if err != nil {
		return "", err
	}
	api := c.api(cfg.token)
	repoType := string(normalizeRepoType(opts.RepoType))
	revision := normalizeRevision(opts.Revision)

	if opts.LocalFilesOnly {
		return c.localSnapshotOnly(opts, cfg.cacheDir, repoType, revision)
	}

	c.report(cfg.progress, ProgressEvent{RepoID: opts.RepoID, State: "resolving"})
	commit, err := api.ResolveRevision(ctx, repoType, opts.RepoID, revision)
	if err != nil {
		return "", wrapHFError(err)
	}
	if commit == "" {
		commit = revision
	}
	if err := cache.WriteRef(cfg.cacheDir, repoType, opts.RepoID, revision, commit); err != nil {
		return "", err
	}

	c.report(cfg.progress, ProgressEvent{RepoID: opts.RepoID, State: "planning"})
	files, err := api.ListRepoTree(ctx, repoType, opts.RepoID, revision)
	if err != nil {
		return "", wrapHFError(err)
	}
	selected := make([]hfhub.RepoFile, 0, len(files))
	for _, file := range files {
		if file.Type == "file" && patterns.Match(file.Path, opts.Include, opts.Exclude) {
			selected = append(selected, file)
		}
	}
	for _, file := range selected {
		c.report(cfg.progress, ProgressEvent{
			RepoID:     opts.RepoID,
			Filename:   file.Path,
			BytesTotal: file.Size,
			FilesTotal: len(selected),
			State:      "planned",
		})
	}
	c.report(cfg.progress, ProgressEvent{
		RepoID:     opts.RepoID,
		FilesTotal: len(selected),
		State:      "plan_complete",
	})

	workers := opts.MaxWorkers
	if workers <= 0 {
		workers = min(8, max(1, runtime.NumCPU()*2))
	}

	var filesDone atomic.Int64
	jobs := make(chan hfhub.RepoFile)
	errs := make(chan error, len(selected))
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for file := range jobs {
				meta := hfhub.FileMetadata{
					Commit: commit,
					ETag:   firstNonEmpty(file.LFSSHA256(), file.BlobID),
					Size:   file.Size,
				}
				if _, err := c.downloadOne(ctx, api, cfg.cacheDir, repoType, opts.RepoID, commit, commit, file.Path, meta, opts.Force, cfg.progress); err != nil {
					errs <- err
					continue
				}
				done := int(filesDone.Add(1))
				c.report(cfg.progress, ProgressEvent{
					RepoID:     opts.RepoID,
					Filename:   file.Path,
					FilesDone:  done,
					FilesTotal: len(selected),
					State:      "file_complete",
				})
			}
		}()
	}
	for _, file := range selected {
		select {
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return "", ctx.Err()
		case jobs <- file:
		}
	}
	close(jobs)
	wg.Wait()
	close(errs)

	var joined error
	for err := range errs {
		joined = errors.Join(joined, err)
	}
	if joined != nil {
		return "", joined
	}

	snapshotRoot := cache.SnapshotDir(cfg.cacheDir, repoType, opts.RepoID, commit)
	if opts.LocalDir != "" {
		for _, file := range selected {
			src, err := cache.SnapshotFilePath(cfg.cacheDir, repoType, opts.RepoID, commit, file.Path)
			if err != nil {
				return "", err
			}
			if _, err := c.materializeFile(ctx, opts.RepoID, file.Path, src, opts.LocalDir, cfg.progress); err != nil {
				return "", err
			}
		}
		snapshotRoot = opts.LocalDir
	}
	c.report(cfg.progress, ProgressEvent{
		RepoID:      opts.RepoID,
		FilesDone:   len(selected),
		FilesTotal:  len(selected),
		State:       "complete",
		Destination: snapshotRoot,
	})
	return snapshotRoot, nil
}

type clientConfig struct {
	cacheDir string
	token    string
	progress ProgressReporter
}

func (c *Client) config(cacheDir, token string, progress ProgressReporter) (clientConfig, error) {
	resolvedCache, err := cache.ResolveCacheDir(firstNonEmpty(cacheDir, c.cacheDir))
	if err != nil {
		return clientConfig{}, err
	}
	resolvedToken, err := resolveToken(firstNonEmpty(token, c.token))
	if err != nil {
		return clientConfig{}, err
	}
	return clientConfig{
		cacheDir: resolvedCache,
		token:    resolvedToken,
		progress: firstProgress(progress, c.progress),
	}, nil
}

func (c *Client) api(token string) *hfhub.Client {
	endpoint := strings.TrimRight(c.endpoint, "/")
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	httpClient := c.httpClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return hfhub.NewClient(endpoint, httpClient, token)
}

func (c *Client) downloadOne(ctx context.Context, api *hfhub.Client, cacheDir, repoType, repoID, commit, revision, filename string, meta hfhub.FileMetadata, force bool, progress ProgressReporter) (string, error) {
	snapshotPath, err := cache.SnapshotFilePath(cacheDir, repoType, repoID, commit, filename)
	if err != nil {
		return "", err
	}
	if !force && cache.FileExists(snapshotPath) {
		c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, BytesTotal: meta.Size, State: "cached", Destination: snapshotPath})
		return snapshotPath, nil
	}

	lock, err := cache.AcquireLock(ctx, cacheDir, repoType, repoID, filename)
	if err != nil {
		return "", err
	}
	defer lock.Release()

	if !force && cache.FileExists(snapshotPath) {
		c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, BytesTotal: meta.Size, State: "cached", Destination: snapshotPath})
		return snapshotPath, nil
	}

	blobKey := cache.BlobKey(meta.ETag, filename)
	blobPath := cache.BlobPath(cacheDir, repoType, repoID, blobKey)
	if force || !cache.FileExists(blobPath) {
		if err := cache.EnsureParent(blobPath); err != nil {
			return "", err
		}
		tmp, err := os.CreateTemp(filepath.Dir(blobPath), ".nego-*")
		if err != nil {
			return "", err
		}
		tmpPath := tmp.Name()
		c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, BytesTotal: meta.Size, State: "downloading"})
		downloadMeta, downloadErr := api.DownloadFile(ctx, repoType, repoID, revision, filename, tmp, func(done, total int64) {
			c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, BytesDone: done, BytesTotal: total, State: "downloading"})
		})
		closeErr := tmp.Close()
		if downloadErr != nil {
			_ = os.Remove(tmpPath)
			return "", wrapHFError(downloadErr)
		}
		if closeErr != nil {
			_ = os.Remove(tmpPath)
			return "", closeErr
		}
		if downloadMeta.Size > 0 {
			meta.Size = downloadMeta.Size
		}
		if err := cache.ReplaceFile(tmpPath, blobPath); err != nil {
			_ = os.Remove(tmpPath)
			return "", err
		}
	}
	c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, BytesTotal: meta.Size, State: "caching"})
	if err := cache.CopyFileWithProgress(ctx, blobPath, snapshotPath, func(done, total int64) {
		if total <= 0 {
			total = meta.Size
		}
		c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, BytesDone: done, BytesTotal: total, State: "caching"})
	}); err != nil {
		return "", err
	}
	return snapshotPath, nil
}

func (c *Client) localFileOnly(opts DownloadFileOptions, cacheDir, repoType, revision string) (string, error) {
	if opts.LocalDir != "" {
		path, err := cache.LocalFilePath(opts.LocalDir, opts.Filename)
		if err != nil {
			return "", err
		}
		if cache.FileExists(path) {
			return path, nil
		}
	}
	commit := revision
	if ref, err := cache.ReadRef(cacheDir, repoType, opts.RepoID, revision); err == nil && ref != "" {
		commit = ref
	}
	path, err := cache.SnapshotFilePath(cacheDir, repoType, opts.RepoID, commit, opts.Filename)
	if err != nil {
		return "", err
	}
	if cache.FileExists(path) {
		return path, nil
	}
	return "", incompleteSnapshot("file %q is not available locally for %s@%s", opts.Filename, opts.RepoID, revision)
}

func (c *Client) localSnapshotOnly(opts DownloadSnapshotOptions, cacheDir, repoType, revision string) (string, error) {
	if opts.LocalDir != "" && cache.DirExists(opts.LocalDir) {
		return opts.LocalDir, nil
	}
	commit := revision
	if ref, err := cache.ReadRef(cacheDir, repoType, opts.RepoID, revision); err == nil && ref != "" {
		commit = ref
	}
	path := cache.SnapshotDir(cacheDir, repoType, opts.RepoID, commit)
	if cache.DirExists(path) {
		return path, nil
	}
	return "", incompleteSnapshot("snapshot is not available locally for %s@%s", opts.RepoID, revision)
}

func (c *Client) finishCachedFile(ctx context.Context, opts DownloadFileOptions, snapshotPath string, progress ProgressReporter, meta *hfhub.FileMetadata) (string, error) {
	path := snapshotPath
	var err error
	if opts.LocalDir != "" {
		path, err = c.materializeFile(ctx, opts.RepoID, opts.Filename, snapshotPath, opts.LocalDir, progress)
		if err != nil {
			return "", err
		}
		c.report(progress, ProgressEvent{RepoID: opts.RepoID, Filename: opts.Filename, BytesTotal: totalFromMeta(meta), State: "file_complete", Destination: path})
		return path, nil
	}
	c.report(progress, ProgressEvent{RepoID: opts.RepoID, Filename: opts.Filename, BytesTotal: totalFromMeta(meta), State: "cached", Destination: path})
	return path, nil
}

func totalFromMeta(meta *hfhub.FileMetadata) int64 {
	if meta == nil {
		return 0
	}
	return meta.Size
}

func (c *Client) materializeFile(ctx context.Context, repoID, filename, src, localDir string, progress ProgressReporter) (string, error) {
	c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, State: "materializing", Destination: localDir})
	path, err := cache.MaterializeFileWithProgress(ctx, src, localDir, filename, func(done, total int64) {
		c.report(progress, ProgressEvent{
			RepoID:      repoID,
			Filename:    filename,
			BytesDone:   done,
			BytesTotal:  total,
			State:       "materializing",
			Destination: localDir,
		})
	})
	if err != nil {
		return "", err
	}
	c.report(progress, ProgressEvent{RepoID: repoID, Filename: filename, State: "materialized", Destination: path})
	return path, nil
}

func (c *Client) report(reporter ProgressReporter, event ProgressEvent) {
	if reporter != nil {
		reporter.ReportProgress(event)
	}
}

func validateFileOptions(opts *DownloadFileOptions) error {
	if err := validateCommon(opts.RepoID, opts.RepoType); err != nil {
		return err
	}
	if strings.TrimSpace(opts.Filename) == "" {
		return invalidOptions("filename is required")
	}
	return cache.ValidateRepoPath(opts.Filename)
}

func validateSnapshotOptions(opts *DownloadSnapshotOptions) error {
	if err := validateCommon(opts.RepoID, opts.RepoType); err != nil {
		return err
	}
	for _, pattern := range append(append([]string{}, opts.Include...), opts.Exclude...) {
		if strings.TrimSpace(pattern) == "" {
			return invalidOptions("include/exclude patterns cannot be empty")
		}
	}
	return nil
}

func validateCommon(repoID string, repoType RepoType) error {
	if strings.Count(strings.TrimSpace(repoID), "/") != 1 {
		return invalidOptions("repo id must look like namespace/name")
	}
	switch normalizeRepoType(repoType) {
	case RepoTypeModel, RepoTypeDataset, RepoTypeSpace:
		return nil
	default:
		return invalidOptions("unsupported repo type %q", repoType)
	}
}

func normalizeRepoType(repoType RepoType) RepoType {
	if repoType == "" {
		return RepoTypeModel
	}
	return repoType
}

func normalizeRevision(revision string) string {
	if strings.TrimSpace(revision) == "" {
		return "main"
	}
	return revision
}

func resolveToken(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	if token := os.Getenv("HF_TOKEN"); token != "" {
		return token, nil
	}
	tokenPath := os.Getenv("HF_TOKEN_PATH")
	if tokenPath == "" {
		hfHome := os.Getenv("HF_HOME")
		if hfHome == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", nil
			}
			hfHome = filepath.Join(home, ".cache", "huggingface")
		}
		tokenPath = filepath.Join(hfHome, "token")
	}
	data, err := os.ReadFile(tokenPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func firstProgress(values ...ProgressReporter) ProgressReporter {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}
