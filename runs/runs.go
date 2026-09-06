package runs

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	nego "github.com/gakon/nego-ai"
)

type Entry struct {
	ID            string            `json:"id"`
	Command       string            `json:"command"`
	Backend       string            `json:"backend,omitempty"`
	Path          string            `json:"path,omitempty"`
	Model         string            `json:"model,omitempty"`
	Endpoint      string            `json:"endpoint,omitempty"`
	Prompt        string            `json:"prompt,omitempty"`
	Messages      []nego.Message    `json:"messages,omitempty"`
	Output        string            `json:"output,omitempty"`
	Error         string            `json:"error,omitempty"`
	StartedAt     time.Time         `json:"started_at"`
	DurationMS    int64             `json:"duration_ms"`
	MaxTokens     int               `json:"max_tokens,omitempty"`
	Temperature   float64           `json:"temperature,omitempty"`
	TopP          float64           `json:"top_p,omitempty"`
	RepeatPenalty float64           `json:"repeat_penalty,omitempty"`
	Stop          []string          `json:"stop,omitempty"`
	Seed          int64             `json:"seed,omitempty"`
	Options       map[string]string `json:"options,omitempty"`
}

func NewID() string {
	var data [8]byte
	if _, err := rand.Read(data[:]); err == nil {
		return hex.EncodeToString(data[:])
	}
	return fmt.Sprintf("%x", time.Now().UnixNano())
}

func Append(path string, entry Entry) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("run log path is required")
	}
	if entry.ID == "" {
		entry.ID = NewID()
	}
	if entry.StartedAt.IsZero() {
		entry.StartedAt = time.Now().UTC()
	}
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	if _, err := file.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}

func Read(path string) ([]Entry, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return Decode(file)
}

func Decode(r io.Reader) ([]Entry, error) {
	var entries []Entry
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 16*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var entry Entry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return entries, nil
}

func Find(entries []Entry, id string) (Entry, bool) {
	for _, entry := range entries {
		if entry.ID == id {
			return entry, true
		}
	}
	return Entry{}, false
}
