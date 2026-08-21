package hfhub

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

type Client struct {
	endpoint   string
	httpClient *http.Client
	token      string
}

func NewClient(endpoint string, httpClient *http.Client, token string) *Client {
	return &Client{
		endpoint:   strings.TrimRight(endpoint, "/"),
		httpClient: httpClient,
		token:      token,
	}
}

type HTTPError struct {
	StatusCode int
	URL        string
	Message    string
	MaybeGated bool
}

func (e *HTTPError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	return fmt.Sprintf("hugging face hub returned HTTP %d for %s", e.StatusCode, e.URL)
}

type RepoFile struct {
	Type   string   `json:"type"`
	Path   string   `json:"path"`
	Size   int64    `json:"size"`
	BlobID string   `json:"blob_id"`
	LFS    *LFSInfo `json:"lfs"`
}

type LFSInfo struct {
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

func (f RepoFile) LFSSHA256() string {
	if f.LFS == nil {
		return ""
	}
	return f.LFS.SHA256
}

type FileMetadata struct {
	Commit string
	ETag   string
	Size   int64
}

func (c *Client) ResolveRevision(ctx context.Context, repoType, repoID, revision string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.apiRepoURL(repoType, repoID, revision), nil)
	if err != nil {
		return "", err
	}
	c.addHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", c.httpError(resp)
	}
	var body struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	return body.SHA, nil
}

func (c *Client) ListRepoTree(ctx context.Context, repoType, repoID, revision string) ([]RepoFile, error) {
	nextURL := c.apiTreeURL(repoType, repoID, revision)
	var out []RepoFile
	for nextURL != "" {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, nextURL, nil)
		if err != nil {
			return nil, err
		}
		c.addHeaders(req)
		resp, err := c.httpClient.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			err := c.httpError(resp)
			resp.Body.Close()
			return nil, err
		}
		var page []RepoFile
		if err := json.NewDecoder(resp.Body).Decode(&page); err != nil {
			resp.Body.Close()
			return nil, err
		}
		nextURL = nextLink(resp.Header.Get("Link"))
		resp.Body.Close()
		out = append(out, page...)
	}
	return out, nil
}

func (c *Client) HeadFile(ctx context.Context, repoType, repoID, revision, filename string) (*FileMetadata, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, c.resolveURL(repoType, repoID, revision, filename), nil)
	if err != nil {
		return nil, err
	}
	c.addHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, c.httpError(resp)
	}
	return metadataFromHeaders(resp.Header), nil
}

func (c *Client) DownloadFile(ctx context.Context, repoType, repoID, revision, filename string, w io.Writer, progress func(done, total int64)) (*FileMetadata, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.resolveURL(repoType, repoID, revision, filename), nil)
	if err != nil {
		return nil, err
	}
	c.addHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, c.httpError(resp)
	}
	meta := metadataFromHeaders(resp.Header)
	var done int64
	buf := make([]byte, 1024*256)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := w.Write(buf[:n]); err != nil {
				return nil, err
			}
			done += int64(n)
			if progress != nil {
				progress(done, meta.Size)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, readErr
		}
	}
	if meta.Size > 0 && done != meta.Size {
		return nil, fmt.Errorf("downloaded %d bytes, expected %d", done, meta.Size)
	}
	return meta, nil
}

func (c *Client) apiRepoURL(repoType, repoID, revision string) string {
	return fmt.Sprintf("%s/api/%s/%s/revision/%s", c.endpoint, apiKind(repoType), escapeRepoID(repoID), url.PathEscape(revision))
}

func (c *Client) apiTreeURL(repoType, repoID, revision string) string {
	values := url.Values{}
	values.Set("recursive", "true")
	values.Set("expand", "false")
	return fmt.Sprintf("%s/api/%s/%s/tree/%s?%s", c.endpoint, apiKind(repoType), escapeRepoID(repoID), url.PathEscape(revision), values.Encode())
}

func (c *Client) resolveURL(repoType, repoID, revision, filename string) string {
	prefix := ""
	switch repoType {
	case "dataset":
		prefix = "datasets/"
	case "space":
		prefix = "spaces/"
	}
	return fmt.Sprintf("%s/%s%s/resolve/%s/%s", c.endpoint, prefix, escapeRepoID(repoID), url.PathEscape(revision), escapeRepoPath(filename))
}

func (c *Client) addHeaders(req *http.Request) {
	req.Header.Set("User-Agent", "nego/0.1")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
}

func (c *Client) httpError(resp *http.Response) error {
	defer io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = fmt.Sprintf("hugging face hub returned HTTP %d", resp.StatusCode)
	}
	lower := strings.ToLower(msg)
	return &HTTPError{
		StatusCode: resp.StatusCode,
		URL:        resp.Request.URL.String(),
		Message:    msg,
		MaybeGated: strings.Contains(lower, "gated") || strings.Contains(lower, "access") || strings.Contains(lower, "authorized"),
	}
}

func metadataFromHeaders(header http.Header) *FileMetadata {
	size := int64(0)
	if raw := header.Get("Content-Length"); raw != "" {
		if parsed, err := strconv.ParseInt(raw, 10, 64); err == nil {
			size = parsed
		}
	}
	return &FileMetadata{
		Commit: header.Get("X-Repo-Commit"),
		ETag:   firstHeader(header, "X-Linked-Etag", "ETag"),
		Size:   size,
	}
}

func apiKind(repoType string) string {
	switch repoType {
	case "dataset":
		return "datasets"
	case "space":
		return "spaces"
	default:
		return "models"
	}
}

func escapeRepoID(repoID string) string {
	parts := strings.Split(repoID, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func escapeRepoPath(repoPath string) string {
	parts := strings.Split(repoPath, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func firstHeader(header http.Header, names ...string) string {
	for _, name := range names {
		if value := strings.Trim(header.Get(name), "\""); value != "" {
			return value
		}
	}
	return ""
}

func nextLink(linkHeader string) string {
	for _, part := range strings.Split(linkHeader, ",") {
		sections := strings.Split(part, ";")
		if len(sections) < 2 {
			continue
		}
		link := strings.TrimSpace(sections[0])
		if !strings.HasPrefix(link, "<") || !strings.HasSuffix(link, ">") {
			continue
		}
		for _, section := range sections[1:] {
			if strings.TrimSpace(section) == `rel="next"` {
				return strings.TrimSuffix(strings.TrimPrefix(link, "<"), ">")
			}
		}
	}
	return ""
}
