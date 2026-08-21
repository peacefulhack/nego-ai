package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/gakon/nego-ai/hub"
	"github.com/gakon/nego-ai/internal/cache"
	"github.com/gakon/nego-ai/internal/registry"
)

func Run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return 2
	}
	switch args[0] {
	case "download":
		return runDownload(ctx, args[1:], stdout, stderr)
	case "models":
		return runModels(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		usage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown command %q\n", args[0])
		usage(stderr)
		return 2
	}
}

type repeatedFlag []string

func (f *repeatedFlag) String() string {
	return strings.Join(*f, ",")
}

func (f *repeatedFlag) Set(value string) error {
	*f = append(*f, value)
	return nil
}

func runDownload(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	var include repeatedFlag
	var exclude repeatedFlag
	var localDir, cacheDir, revision, token, repoType string
	var force, localOnly, quiet, jsonOutput bool
	var workers int

	fs := flag.NewFlagSet("download", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&localDir, "local-dir", "", "download files into a local directory")
	fs.StringVar(&cacheDir, "cache-dir", "", "cache directory")
	fs.StringVar(&revision, "revision", "main", "branch, tag, pull request ref, or commit")
	fs.StringVar(&token, "token", "", "Hugging Face token")
	fs.StringVar(&repoType, "repo-type", "model", "repo type: model, dataset, or space")
	fs.BoolVar(&force, "force", false, "force re-download")
	fs.BoolVar(&localOnly, "local-files-only", false, "only use local files")
	fs.BoolVar(&quiet, "quiet", false, "disable progress output")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	fs.IntVar(&workers, "workers", 0, "maximum concurrent snapshot downloads")
	fs.Var(&include, "include", "include glob pattern, repeatable")
	fs.Var(&exclude, "exclude", "exclude glob pattern, repeatable")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) < 1 || len(positionals) > 2 {
		fmt.Fprintln(stderr, "usage: nego download <repo-id> [filename] [flags]")
		return 2
	}

	var reporter hub.ProgressReporter
	if !quiet {
		reporter = &terminalProgress{w: stderr}
	}
	if jsonOutput && !quiet {
		reporter = &jsonProgress{w: stderr}
	}

	client := hub.NewClient()
	commonRepoType := hub.RepoType(repoType)
	var path string
	var err error
	if len(positionals) == 2 {
		path, err = client.DownloadFile(ctx, hub.DownloadFileOptions{
			RepoID:         positionals[0],
			Filename:       positionals[1],
			RepoType:       commonRepoType,
			Revision:       revision,
			LocalDir:       localDir,
			CacheDir:       cacheDir,
			Token:          token,
			Force:          force,
			LocalFilesOnly: localOnly,
			Progress:       reporter,
		})
	} else {
		path, err = client.DownloadSnapshot(ctx, hub.DownloadSnapshotOptions{
			RepoID:         positionals[0],
			RepoType:       commonRepoType,
			Revision:       revision,
			LocalDir:       localDir,
			CacheDir:       cacheDir,
			Token:          token,
			Include:        include,
			Exclude:        exclude,
			Force:          force,
			LocalFilesOnly: localOnly,
			MaxWorkers:     workers,
			Progress:       reporter,
		})
	}
	if reporterWithFinish, ok := reporter.(interface{ Finish() }); ok {
		reporterWithFinish.Finish()
	}
	if err != nil {
		printError(stderr, err)
		return exitCode(err)
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(map[string]string{"path": path})
	} else if quiet {
		fmt.Fprintln(stdout, path)
	} else {
		fmt.Fprintf(stdout, "Saved to: %s\n", path)
	}
	return 0
}

func splitFlags(args []string) ([]string, []string) {
	var flags []string
	var positionals []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			positionals = append(positionals, args[i+1:]...)
			break
		}
		if strings.HasPrefix(arg, "-") && arg != "-" {
			flags = append(flags, arg)
			if strings.Contains(arg, "=") || isBoolFlag(arg) {
				continue
			}
			if i+1 < len(args) {
				i++
				flags = append(flags, args[i])
			}
			continue
		}
		positionals = append(positionals, arg)
	}
	return flags, positionals
}

func isBoolFlag(arg string) bool {
	name := strings.TrimLeft(arg, "-")
	switch name {
	case "force", "local-files-only", "quiet", "json":
		return true
	default:
		return false
	}
}

func printError(w io.Writer, err error) {
	var hubErr *hub.Error
	if errors.As(err, &hubErr) {
		fmt.Fprintf(w, "nego: %s\n", hubErr.Message)
		if hubErr.Kind == hub.ErrGated || hubErr.Kind == hub.ErrUnauthorized || hubErr.Kind == hub.ErrForbidden {
			fmt.Fprintln(w, "nego: pass --token or set HF_TOKEN; gated models also require access approval on Hugging Face")
		}
		return
	}
	fmt.Fprintf(w, "nego: %v\n", err)
}

func exitCode(err error) int {
	var hubErr *hub.Error
	if !errors.As(err, &hubErr) {
		return 1
	}
	switch hubErr.Kind {
	case hub.ErrInvalidOptions:
		return 2
	case hub.ErrNotFound, hub.ErrRevisionNotFound:
		return 4
	case hub.ErrUnauthorized, hub.ErrForbidden, hub.ErrGated:
		return 5
	case hub.ErrRateLimited:
		return 6
	default:
		return 1
	}
}

func usage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego download <repo-id> [filename] [flags]")
	fmt.Fprintln(w, "  nego models list [flags]")
	fmt.Fprintln(w, "  nego models info <repo-id> [flags]")
}

func runModels(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		modelsUsage(stderr)
		return 2
	}
	switch args[0] {
	case "list":
		return runModelsList(args[1:], stdout, stderr)
	case "info":
		return runModelsInfo(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		modelsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown models command %q\n", args[0])
		modelsUsage(stderr)
		return 2
	}
}

func runModelsInfo(args []string, stdout, stderr io.Writer) int {
	var cacheDir string
	var revision string
	var jsonOutput bool

	fs := flag.NewFlagSet("models info", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cacheDir, "cache-dir", "", "cache directory")
	fs.StringVar(&revision, "revision", "", "revision to inspect")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego models info <repo-id> [flags]")
		return 2
	}

	resolvedCache, err := cache.ResolveCacheDir(cacheDir)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	entry, err := registry.NewStore(resolvedCache).Find(positionals[0], revision)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(entry)
		return 0
	}
	writeModelInfo(stdout, entry)
	return 0
}

func runModelsList(args []string, stdout, stderr io.Writer) int {
	var cacheDir string
	var jsonOutput bool

	fs := flag.NewFlagSet("models list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cacheDir, "cache-dir", "", "cache directory")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: nego models list [flags]")
		return 2
	}

	resolvedCache, err := cache.ResolveCacheDir(cacheDir)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	entries, err := registry.NewStore(resolvedCache).List()
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(entries)
		return 0
	}
	if len(entries) == 0 {
		fmt.Fprintln(stdout, "No local models found.")
		return 0
	}

	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "REPO\tREVISION\tSIZE\tFILES\tLOCAL PATH")
	for _, entry := range entries {
		localPath := entry.LocalDir
		if localPath == "" {
			localPath = entry.SnapshotPath
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\n", entry.RepoID, entry.Revision, humanBytes(entry.TotalSize), entry.FileCount, localPath)
	}
	_ = tw.Flush()
	return 0
}

func modelsUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego models list [flags]")
	fmt.Fprintln(w, "  nego models info <repo-id> [flags]")
}

func writeModelInfo(w io.Writer, entry registry.Entry) {
	localPath := entry.LocalDir
	if localPath == "" {
		localPath = entry.SnapshotPath
	}
	fmt.Fprintf(w, "Repo:        %s\n", entry.RepoID)
	fmt.Fprintf(w, "Type:        %s\n", entry.RepoType)
	fmt.Fprintf(w, "Revision:    %s\n", entry.Revision)
	fmt.Fprintf(w, "Commit:      %s\n", entry.Commit)
	fmt.Fprintf(w, "Local path:  %s\n", localPath)
	fmt.Fprintf(w, "Cache path:  %s\n", entry.SnapshotPath)
	fmt.Fprintf(w, "Files:       %d\n", entry.FileCount)
	fmt.Fprintf(w, "Size:        %s\n", humanBytes(entry.TotalSize))
	if !entry.DownloadedAt.IsZero() {
		fmt.Fprintf(w, "Downloaded:  %s\n", entry.DownloadedAt.Format(time.RFC3339))
	}
	keyFiles := importantFiles(entry.Files)
	if len(keyFiles) > 0 {
		fmt.Fprintln(w, "Key files:")
		for _, file := range keyFiles {
			fmt.Fprintf(w, "  - %s", file.Path)
			if file.Size > 0 {
				fmt.Fprintf(w, " (%s)", humanBytes(file.Size))
			}
			fmt.Fprintln(w)
		}
	}
}

func importantFiles(files []registry.File) []registry.File {
	var out []registry.File
	for _, file := range files {
		if isImportantModelFile(file.Path) {
			out = append(out, file)
		}
	}
	return out
}

func isImportantModelFile(path string) bool {
	lower := strings.ToLower(path)
	switch lower {
	case "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json":
		return true
	default:
		return strings.HasSuffix(lower, ".safetensors") || strings.HasSuffix(lower, ".gguf") || strings.HasSuffix(lower, ".onnx")
	}
}

type terminalProgress struct {
	mu            sync.Mutex
	w             io.Writer
	lastRender    time.Time
	entries       map[string]*progressEntry
	order         []string
	title         string
	renderedLines int
	wrote         bool
	tick          int
	stopTicker    chan struct{}
	tickerDone    chan struct{}
}

type progressEntry struct {
	name       string
	state      string
	bytesDone  int64
	bytesTotal int64
	started    time.Time
}

func (p *terminalProgress) ReportProgress(event hub.ProgressEvent) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.startTickerLocked()
	if p.entries == nil {
		p.entries = make(map[string]*progressEntry)
	}

	force := false
	switch event.State {
	case "resolving":
		p.title = "Resolving..."
		force = true
	case "planning":
		p.title = "Planning downloads..."
		force = true
	case "planned":
		entry := p.entry(event.Filename)
		entry.state = "Queued"
		entry.bytesTotal = event.BytesTotal
		if event.FilesTotal == 1 {
			p.title = "Downloading 1 file"
			force = true
		}
	case "plan_complete":
		p.title = fmt.Sprintf("Downloading %d files", event.FilesTotal)
		if event.FilesTotal == 1 {
			p.title = "Downloading 1 file"
		}
		force = true
	case "downloading":
		entry := p.entry(event.Filename)
		entry.state = "Downloading"
		entry.bytesDone = event.BytesDone
		entry.bytesTotal = event.BytesTotal
		if entry.started.IsZero() {
			entry.started = time.Now()
		}
	case "caching":
		entry := p.entry(event.Filename)
		entry.state = "Caching"
		entry.bytesDone = event.BytesDone
		entry.bytesTotal = event.BytesTotal
		if entry.started.IsZero() {
			entry.started = time.Now()
		}
	case "materializing":
		entry := p.entry(event.Filename)
		entry.state = "Materializing"
		entry.bytesDone = event.BytesDone
		entry.bytesTotal = event.BytesTotal
		if entry.started.IsZero() {
			entry.started = time.Now()
		}
	case "materialized":
		entry := p.entry(event.Filename)
		entry.state = "Done"
		if entry.bytesTotal > 0 {
			entry.bytesDone = entry.bytesTotal
		}
		force = true
	case "cached":
		entry := p.entry(event.Filename)
		entry.state = "Cached"
		entry.bytesDone = event.BytesTotal
		entry.bytesTotal = event.BytesTotal
		force = true
	case "file_complete":
		entry := p.entry(event.Filename)
		if entry.state != "Cached" {
			entry.state = "Done"
		}
		if entry.bytesTotal > 0 {
			entry.bytesDone = entry.bytesTotal
		}
		force = true
	case "complete":
		p.title = "Done"
		if event.Destination != "" {
			p.title += ". Saved to " + event.Destination
		}
		force = true
	}
	p.render(force)
}

func (p *terminalProgress) Finish() {
	var done chan struct{}
	p.mu.Lock()
	if p.stopTicker != nil {
		close(p.stopTicker)
		done = p.tickerDone
		p.stopTicker = nil
	}
	p.mu.Unlock()
	if done != nil {
		<-done
	}
}

func (p *terminalProgress) startTickerLocked() {
	if p.stopTicker != nil {
		return
	}
	p.stopTicker = make(chan struct{})
	p.tickerDone = make(chan struct{})
	stop := p.stopTicker
	done := p.tickerDone
	go func() {
		ticker := time.NewTicker(120 * time.Millisecond)
		defer ticker.Stop()
		defer close(done)
		for {
			select {
			case <-ticker.C:
				p.mu.Lock()
				p.render(false)
				p.mu.Unlock()
			case <-stop:
				return
			}
		}
	}()
}

func (p *terminalProgress) entry(filename string) *progressEntry {
	if filename == "" {
		filename = "(unknown)"
	}
	entry := p.entries[filename]
	if entry != nil {
		return entry
	}
	entry = &progressEntry{name: filename, state: "Queued"}
	p.entries[filename] = entry
	p.order = append(p.order, filename)
	return entry
}

func (p *terminalProgress) render(force bool) {
	if !force && time.Since(p.lastRender) < 120*time.Millisecond {
		return
	}
	lines := renderProgressLines(p.title, p.order, p.entries, spinnerFrame(p.tick))
	if len(lines) == 0 {
		return
	}
	if p.renderedLines > 0 {
		fmt.Fprintf(p.w, "\033[%dA", p.renderedLines)
	}
	totalLines := max(p.renderedLines, len(lines))
	for i := 0; i < totalLines; i++ {
		fmt.Fprint(p.w, "\r\033[2K")
		if i < len(lines) {
			fmt.Fprint(p.w, lines[i])
		}
		fmt.Fprint(p.w, "\n")
	}
	p.renderedLines = totalLines
	p.lastRender = time.Now()
	p.wrote = true
	p.tick++
}

type jsonProgress struct {
	mu sync.Mutex
	w  io.Writer
}

func (p *jsonProgress) ReportProgress(event hub.ProgressEvent) {
	p.mu.Lock()
	defer p.mu.Unlock()
	_ = json.NewEncoder(p.w).Encode(event)
}

func (p *jsonProgress) Finish() {}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func renderProgressLines(title string, order []string, entries map[string]*progressEntry, spinner string) []string {
	if title == "" && len(order) == 0 {
		return nil
	}
	if title == "" {
		title = "Downloading"
	}
	lines := []string{spinner + " " + title}
	for _, filename := range order {
		entry := entries[filename]
		if entry == nil {
			continue
		}
		lines = append(lines, renderProgressEntry(entry, spinner))
	}
	return lines
}

func renderProgressEntry(entry *progressEntry, spinner string) string {
	name := shorten(entry.name, 42)
	switch entry.state {
	case "Downloading", "Caching", "Materializing":
		bar := progressBar(entry.bytesDone, entry.bytesTotal, 18)
		speed := bytesPerSecond(entry.bytesDone, time.Since(entry.started))
		if entry.bytesTotal > 0 {
			percent := float64(entry.bytesDone) / float64(entry.bytesTotal) * 100
			if percent > 100 {
				percent = 100
			}
			return fmt.Sprintf("  %s %-42s %-13s %s %6.2f%% %s/%s %s/s", spinner, name, entry.state, bar, percent, humanBytes(entry.bytesDone), humanBytes(entry.bytesTotal), humanBytes(speed))
		}
		return fmt.Sprintf("  %s %-42s %-13s %s %s %s/s", spinner, name, entry.state, bar, humanBytes(entry.bytesDone), humanBytes(speed))
	case "Done":
		return fmt.Sprintf("  ✓ %-42s Done          %s", name, doneSize(entry))
	case "Cached":
		return fmt.Sprintf("  ✓ %-42s Cached        %s", name, doneSize(entry))
	default:
		return fmt.Sprintf("  • %-42s Queued        %s", name, doneSize(entry))
	}
}

func renderDownloadLine(event hub.ProgressEvent, started time.Time) string {
	entry := &progressEntry{
		name:       event.Filename,
		state:      "Downloading",
		bytesDone:  event.BytesDone,
		bytesTotal: event.BytesTotal,
		started:    started,
	}
	return strings.TrimSpace(renderProgressEntry(entry, spinnerFrame(0)))
}

func doneSize(entry *progressEntry) string {
	if entry.bytesTotal > 0 {
		return humanBytes(entry.bytesTotal)
	}
	return ""
}

func renderFilesLine(event hub.ProgressEvent) string {
	if event.FilesTotal <= 0 {
		return "Fetching files..."
	}
	bar := progressBar(int64(event.FilesDone), int64(event.FilesTotal), 24)
	return fmt.Sprintf("Fetching files %s %d/%d", bar, event.FilesDone, event.FilesTotal)
}

func progressBar(done, total int64, width int) string {
	if width <= 0 {
		width = 24
	}
	if total <= 0 {
		return "[" + strings.Repeat("=", width/2) + ">" + strings.Repeat(".", width-width/2-1) + "]"
	}
	filled := int(float64(done) / float64(total) * float64(width))
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}
	if filled == width {
		return "[" + strings.Repeat("=", width) + "]"
	}
	return "[" + strings.Repeat("=", filled) + ">" + strings.Repeat(".", width-filled-1) + "]"
}

func bytesPerSecond(done int64, elapsed time.Duration) int64 {
	if done <= 0 || elapsed <= 0 {
		return 0
	}
	return int64(float64(done) / elapsed.Seconds())
}

func shorten(value string, maxLen int) string {
	if maxLen <= 0 || len(value) <= maxLen {
		return value
	}
	if maxLen <= 3 {
		return value[:maxLen]
	}
	keep := maxLen - 3
	left := keep / 2
	right := keep - left
	return value[:left] + "..." + value[len(value)-right:]
}

func spinnerFrame(tick int) string {
	frames := []string{"—", "\\", "|", "/"}
	if tick < 0 {
		tick = 0
	}
	return frames[tick%len(frames)]
}
