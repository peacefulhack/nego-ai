package hub

import (
	"context"
	"net/http"
)

const defaultEndpoint = "https://huggingface.co"

type RepoType string

const (
	RepoTypeModel   RepoType = "model"
	RepoTypeDataset RepoType = "dataset"
	RepoTypeSpace   RepoType = "space"
)

type Client struct {
	endpoint   string
	httpClient *http.Client
	token      string
	cacheDir   string
	progress   ProgressReporter
}

type ClientOption func(*Client)

func WithEndpoint(endpoint string) ClientOption {
	return func(c *Client) {
		c.endpoint = endpoint
	}
}

func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *Client) {
		if client != nil {
			c.httpClient = client
		}
	}
}

func WithToken(token string) ClientOption {
	return func(c *Client) {
		c.token = token
	}
}

func WithCacheDir(dir string) ClientOption {
	return func(c *Client) {
		c.cacheDir = dir
	}
}

func WithProgress(reporter ProgressReporter) ClientOption {
	return func(c *Client) {
		c.progress = reporter
	}
}

func NewClient(opts ...ClientOption) *Client {
	c := &Client{
		endpoint:   defaultEndpoint,
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

type DownloadFileOptions struct {
	RepoID         string
	RepoType       RepoType
	Revision       string
	Filename       string
	LocalDir       string
	CacheDir       string
	Token          string
	Force          bool
	LocalFilesOnly bool
	Progress       ProgressReporter
}

type DownloadSnapshotOptions struct {
	RepoID         string
	RepoType       RepoType
	Revision       string
	LocalDir       string
	CacheDir       string
	Token          string
	Include        []string
	Exclude        []string
	Force          bool
	LocalFilesOnly bool
	MaxWorkers     int
	Progress       ProgressReporter
}

type ListFilesOptions struct {
	RepoID   string
	RepoType RepoType
	Revision string
	Token    string
}

type FileInfo struct {
	Path string
	Type string
	Size int64
}

type ResolveGGUFFileOptions struct {
	RepoID   string
	RepoType RepoType
	Revision string
	Token    string
	GGUFRepo string
	Filename string
	Quant    string
}

type GGUFFile struct {
	RepoID   string
	Filename string
	Size     int64
}

type ProgressEvent struct {
	RepoID      string
	Filename    string
	BytesDone   int64
	BytesTotal  int64
	FilesDone   int
	FilesTotal  int
	State       string
	Destination string
}

type ProgressReporter interface {
	ReportProgress(ProgressEvent)
}

func DownloadFile(ctx context.Context, opts DownloadFileOptions) (string, error) {
	return NewClient().DownloadFile(ctx, opts)
}

func DownloadSnapshot(ctx context.Context, opts DownloadSnapshotOptions) (string, error) {
	return NewClient().DownloadSnapshot(ctx, opts)
}

func ListFiles(ctx context.Context, opts ListFilesOptions) ([]FileInfo, error) {
	return NewClient().ListFiles(ctx, opts)
}

func ResolveGGUFFile(ctx context.Context, opts ResolveGGUFFileOptions) (*GGUFFile, error) {
	return NewClient().ResolveGGUFFile(ctx, opts)
}
