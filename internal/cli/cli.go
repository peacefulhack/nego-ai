package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	nego "github.com/gakon/nego-ai"
	_ "github.com/gakon/nego-ai/backends/llama"
	_ "github.com/gakon/nego-ai/backends/native"
	_ "github.com/gakon/nego-ai/backends/openai"
	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/convert"
	"github.com/gakon/nego-ai/datasets"
	"github.com/gakon/nego-ai/evals"
	"github.com/gakon/nego-ai/hub"
	"github.com/gakon/nego-ai/internal/cache"
	"github.com/gakon/nego-ai/internal/registry"
	"github.com/gakon/nego-ai/internal/version"
	"github.com/gakon/nego-ai/modelinfo"
	"github.com/gakon/nego-ai/runs"
	"github.com/gakon/nego-ai/server"
	"github.com/gakon/nego-ai/share"
	"github.com/gakon/nego-ai/tokenizer"
	"github.com/gakon/nego-ai/training"
)

func Run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	return RunWithIO(ctx, args, os.Stdin, stdout, stderr)
}

func RunWithIO(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return 2
	}
	switch args[0] {
	case "download":
		return runDownload(ctx, args[1:], stdout, stderr)
	case "models":
		return runModels(args[1:], stdout, stderr)
	case "runs":
		return runRuns(args[1:], stdout, stderr)
	case "backends":
		return runBackends(args[1:], stdout, stderr)
	case "dataset":
		return runDataset(args[1:], stdout, stderr)
	case "cache":
		return runCache(args[1:], stdout, stderr)
	case "tokenize":
		return runTokenize(args[1:], stdout, stderr)
	case "tokens":
		return runTokens(args[1:], stdout, stderr)
	case "context":
		return runContext(args[1:], stdout, stderr)
	case "prompt":
		return runPrompt(args[1:], stdout, stderr)
	case "inspect":
		return runInspect(args[1:], stdout, stderr)
	case "check":
		return runCheck(args[1:], stdout, stderr)
	case "run":
		return runModel(args[1:], stdout, stderr)
	case "chat":
		return runChat(args[1:], stdin, stdout, stderr)
	case "serve":
		return runServe(args[1:], stdout, stderr)
	case "embed":
		return runEmbed(args[1:], stdout, stderr)
	case "eval":
		return runEval(args[1:], stdout, stderr)
	case "version":
		return runVersion(args[1:], stdout, stderr)
	case "convert":
		return runConvert(args[1:], stdout, stderr)
	case "train":
		return runTrain(args[1:], stdout, stderr)
	case "share":
		return runShare(ctx, args[1:], stdout, stderr)
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
	var ggufRepo, ggufFile, quant string
	var force, localOnly, quiet, jsonOutput bool
	var gguf bool
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
	fs.BoolVar(&gguf, "gguf", false, "download a llama.cpp-ready GGUF file")
	fs.StringVar(&ggufRepo, "gguf-repo", "", "GGUF repo id to use instead of auto-detecting same-owner -GGUF")
	fs.StringVar(&ggufFile, "gguf-file", "", "specific GGUF filename to download")
	fs.StringVar(&quant, "quant", "Q4_K_M", "GGUF quantization to prefer")
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
	var resolvedGGUF *hub.GGUFFile
	if ggufRepo != "" || ggufFile != "" || flagWasSet(fs, "quant") {
		gguf = true
	}
	if gguf {
		if len(include) > 0 || len(exclude) > 0 {
			fmt.Fprintln(stderr, "nego: --gguf cannot be combined with --include or --exclude; use --gguf-file or --quant")
			return 2
		}
		if commonRepoType != hub.RepoTypeModel {
			fmt.Fprintln(stderr, "nego: --gguf only supports model repos")
			return 2
		}
		if len(positionals) == 2 {
			if ggufFile != "" && ggufFile != positionals[1] {
				fmt.Fprintln(stderr, "nego: pass either [filename] or --gguf-file, not both")
				return 2
			}
			ggufFile = positionals[1]
		}
		if localOnly && ggufFile == "" {
			fmt.Fprintln(stderr, "nego: --local-files-only with --gguf requires [filename] or --gguf-file")
			return 2
		}
		if reporter != nil {
			reporter.ReportProgress(hub.ProgressEvent{RepoID: positionals[0], State: "resolving"})
		}
		resolvedGGUF, err = client.ResolveGGUFFile(ctx, hub.ResolveGGUFFileOptions{
			RepoID:   positionals[0],
			RepoType: commonRepoType,
			Revision: revision,
			Token:    token,
			GGUFRepo: ggufRepo,
			Filename: ggufFile,
			Quant:    quant,
		})
		if err == nil {
			path, err = client.DownloadFile(ctx, hub.DownloadFileOptions{
				RepoID:         resolvedGGUF.RepoID,
				Filename:       resolvedGGUF.Filename,
				RepoType:       commonRepoType,
				Revision:       revision,
				LocalDir:       localDir,
				CacheDir:       cacheDir,
				Token:          token,
				Force:          force,
				LocalFilesOnly: localOnly,
				Progress:       reporter,
			})
		}
	} else if len(positionals) == 2 {
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
	warning := ""
	if !gguf && commonRepoType == hub.RepoTypeModel {
		warning = downloadRuntimeWarning(path)
	}
	if jsonOutput {
		result := map[string]string{"path": path}
		if resolvedGGUF != nil {
			result["repo"] = resolvedGGUF.RepoID
			result["file"] = resolvedGGUF.Filename
		}
		if warning != "" {
			result["warning"] = warning
		}
		_ = json.NewEncoder(stdout).Encode(result)
	} else if quiet {
		fmt.Fprintln(stdout, path)
	} else {
		fmt.Fprintf(stdout, "Saved to: %s\n", path)
		if resolvedGGUF != nil {
			fmt.Fprintf(stdout, "Runtime file: %s/%s\n", resolvedGGUF.RepoID, resolvedGGUF.Filename)
		}
		if warning != "" {
			fmt.Fprintf(stderr, "nego: %s\n", warning)
		}
	}
	return 0
}

func downloadRuntimeWarning(path string) string {
	hasSafeTensors, hasGGUF := pathHasRuntimeExtension(path, ".safetensors"), pathHasRuntimeExtension(path, ".gguf")
	if !hasSafeTensors || hasGGUF {
		return ""
	}
	return "downloaded Hugging Face safetensors; Nego cannot run this directly with the local llama.cpp backend. Use `nego download <repo-id> --gguf --local-dir <dir>` for chat/runtime, or `nego convert gguf <model-dir> --out <file>` with a llama.cpp converter."
}

func pathHasRuntimeExtension(root, ext string) bool {
	if root == "" {
		return false
	}
	info, err := os.Stat(root)
	if err != nil {
		return false
	}
	ext = strings.ToLower(ext)
	if !info.IsDir() {
		return strings.HasSuffix(strings.ToLower(root), ext)
	}
	found := false
	_ = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(entry.Name()), ext) {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	return found
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
	case "force", "local-files-only", "quiet", "json", "yes", "no-generation-prompt", "interactive", "flash-attn", "gguf", "native", "no-manifest":
		return true
	default:
		return false
	}
}

func flagWasSet(fs *flag.FlagSet, name string) bool {
	wasSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			wasSet = true
		}
	})
	return wasSet
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
	fmt.Fprintln(w, "  nego download <repo-id> --gguf [--quant Q4_K_M] [--gguf-repo <repo-id>] [flags]")
	fmt.Fprintln(w, "  nego models list [flags]")
	fmt.Fprintln(w, "  nego models info <repo-id> [flags]")
	fmt.Fprintln(w, "  nego models remove <repo-id> --yes [flags]")
	fmt.Fprintln(w, "  nego runs list <runs.jsonl> [flags]")
	fmt.Fprintln(w, "  nego runs show <runs.jsonl> <id> [flags]")
	fmt.Fprintln(w, "  nego backends list [flags]")
	fmt.Fprintln(w, "  nego backends info <name> [flags]")
	fmt.Fprintln(w, "  nego dataset inspect <file> [flags]")
	fmt.Fprintln(w, "  nego dataset validate <file> [flags]")
	fmt.Fprintln(w, "  nego dataset convert <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset filter <file> --where <expr> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset split <file> --train-out <file> --test-out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset sample <file> [flags]")
	fmt.Fprintln(w, "  nego cache usage [flags]")
	fmt.Fprintln(w, "  nego cache gc --yes [flags]")
	fmt.Fprintln(w, "  nego tokenize <model-path> <text> [flags]")
	fmt.Fprintln(w, "  nego tokens <model-path> <text>")
	fmt.Fprintln(w, "  nego context <model-path> <text> [flags]")
	fmt.Fprintln(w, "  nego prompt <model-path> --user <text> [flags]")
	fmt.Fprintln(w, "  nego inspect <model-path> [flags]")
	fmt.Fprintln(w, "  nego check <model-path> [flags]")
	fmt.Fprintln(w, "  nego run <model-path> <prompt> [flags]")
	fmt.Fprintln(w, "  nego chat <model-path> <message> [flags]")
	fmt.Fprintln(w, "  nego serve <model-path> [flags]")
	fmt.Fprintln(w, "  nego embed <text> --endpoint <url> --model <name> [flags]")
	fmt.Fprintln(w, "  nego eval <suite.json> [flags]")
	fmt.Fprintln(w, "  nego eval report <report.json> [flags]")
	fmt.Fprintln(w, "  nego eval compare <baseline.json> <candidate.json> [flags]")
	fmt.Fprintln(w, "  nego version [flags]")
	fmt.Fprintln(w, "  nego convert gguf <model-dir> --out <file> [--converter <path>] [--python <path>]")
	fmt.Fprintln(w, "  nego train init --base-model <dir> --train-file <file> --out <job.json> [flags]")
	fmt.Fprintln(w, "  nego train check <job.json> [flags]")
	fmt.Fprintln(w, "  nego train validate <job.json> [flags]")
	fmt.Fprintln(w, "  nego train native <model-path> --train-file <file> --out <dir> [flags]")
	fmt.Fprintln(w, "  nego train <job.json> [flags]")
	fmt.Fprintln(w, "  nego share manifest <model-dir> --out <file> [flags]")
	fmt.Fprintln(w, "  nego share package <model-dir> --out <archive.tar.gz> [flags]")
	fmt.Fprintln(w, "  nego share upload <repo-id> <local-path> [path-in-repo] [flags]")
}

func runShare(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		shareUsage(stderr)
		return 2
	}
	switch args[0] {
	case "manifest":
		return runShareManifest(args[1:], stdout, stderr)
	case "package":
		return runSharePackage(args[1:], stdout, stderr)
	case "upload":
		return runShareUpload(ctx, args[1:], stdout, stderr)
	case "help", "-h", "--help":
		shareUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown share command %q\n", args[0])
		shareUsage(stderr)
		return 2
	}
}

func runShareManifest(args []string, stdout, stderr io.Writer) int {
	var out string
	var repo string
	var baseModel string
	var jsonOutput bool
	fs := flag.NewFlagSet("share manifest", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&out, "out", "", "manifest output path")
	fs.StringVar(&repo, "repo", "", "target repository id")
	fs.StringVar(&baseModel, "base-model", "", "base model id")
	fs.BoolVar(&jsonOutput, "json", false, "write manifest JSON to stdout")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego share manifest <model-dir> --out <file> [flags]")
		return 2
	}
	manifest, err := share.BuildManifest(share.ManifestOptions{
		Path:      positionals[0],
		RepoID:    repo,
		BaseModel: baseModel,
		Exclude:   []string{out},
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if out != "" {
		if err := share.WriteManifest(out, manifest); err != nil {
			fmt.Fprintf(stderr, "nego: %v\n", err)
			return 1
		}
	}
	if jsonOutput || out == "" {
		_ = json.NewEncoder(stdout).Encode(manifest)
		return 0
	}
	fmt.Fprintf(stdout, "Share manifest: %s\n", out)
	fmt.Fprintf(stdout, "Files:          %d\n", len(manifest.Files))
	fmt.Fprintf(stdout, "Size:           %s\n", humanBytes(manifest.TotalSize))
	return 0
}

func runSharePackage(args []string, stdout, stderr io.Writer) int {
	var out string
	var repo string
	var baseModel string
	var noManifest bool
	fs := flag.NewFlagSet("share package", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&out, "out", "", "archive output path")
	fs.StringVar(&repo, "repo", "", "target repository id")
	fs.StringVar(&baseModel, "base-model", "", "base model id")
	fs.BoolVar(&noManifest, "no-manifest", false, "do not include generated manifest in the archive")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || out == "" {
		fmt.Fprintln(stderr, "usage: nego share package <model-dir> --out <archive.tar.gz> [flags]")
		return 2
	}
	manifest, err := share.PackageArchive(share.PackageOptions{
		Path:            positionals[0],
		Output:          out,
		RepoID:          repo,
		BaseModel:       baseModel,
		IncludeManifest: !noManifest,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Share package:  %s\n", out)
	fmt.Fprintf(stdout, "Files:          %d\n", len(manifest.Files))
	fmt.Fprintf(stdout, "Size:           %s\n", humanBytes(manifest.TotalSize))
	return 0
}

func runShareUpload(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	var repoType string
	var revision string
	var token string
	var endpoint string
	var message string
	var description string
	var createPR bool
	var maxInlineSize int64
	var jsonOutput bool
	var include repeatedFlag
	var exclude repeatedFlag
	fs := flag.NewFlagSet("share upload", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&repoType, "repo-type", "model", "repo type: model, dataset, or space")
	fs.StringVar(&revision, "revision", "main", "target branch or revision")
	fs.StringVar(&token, "token", "", "Hugging Face token")
	fs.StringVar(&endpoint, "endpoint", "", "Hugging Face Hub endpoint")
	fs.StringVar(&message, "message", "", "commit message")
	fs.StringVar(&description, "description", "", "commit description")
	fs.BoolVar(&createPR, "create-pr", false, "open a pull request instead of committing directly")
	fs.Int64Var(&maxInlineSize, "max-inline-size", 0, "maximum inline file size in bytes")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	fs.Var(&include, "include", "include glob pattern for folder upload, repeatable")
	fs.Var(&exclude, "exclude", "exclude glob pattern for folder upload, repeatable")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) < 2 || len(positionals) > 3 {
		fmt.Fprintln(stderr, "usage: nego share upload <repo-id> <local-path> [path-in-repo] [flags]")
		return 2
	}
	repoID := positionals[0]
	localPath := positionals[1]
	pathInRepo := ""
	if len(positionals) == 3 {
		pathInRepo = positionals[2]
	}
	clientOpts := []hub.ClientOption{}
	if endpoint != "" {
		clientOpts = append(clientOpts, hub.WithEndpoint(endpoint))
	}
	client := hub.NewClient(clientOpts...)
	info, err := os.Stat(localPath)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	var result *hub.UploadResult
	if info.IsDir() {
		result, err = client.UploadFolder(ctx, hub.UploadFolderOptions{
			RepoID:            repoID,
			RepoType:          hub.RepoType(repoType),
			Revision:          revision,
			LocalDir:          localPath,
			PathInRepo:        pathInRepo,
			Token:             token,
			Include:           include,
			Exclude:           exclude,
			CommitMessage:     message,
			CommitDescription: description,
			CreatePR:          createPR,
			MaxInlineSize:     maxInlineSize,
		})
	} else {
		result, err = client.UploadFile(ctx, hub.UploadFileOptions{
			RepoID:            repoID,
			RepoType:          hub.RepoType(repoType),
			Revision:          revision,
			LocalPath:         localPath,
			PathInRepo:        pathInRepo,
			Token:             token,
			CommitMessage:     message,
			CommitDescription: description,
			CreatePR:          createPR,
			MaxInlineSize:     maxInlineSize,
		})
	}
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(result)
		return 0
	}
	fmt.Fprintf(stdout, "Uploaded:       %s@%s\n", result.RepoID, result.Revision)
	fmt.Fprintf(stdout, "Files:          %d\n", len(result.Files))
	fmt.Fprintf(stdout, "Size:           %s\n", humanBytes(result.TotalSize))
	if result.Commit != "" {
		fmt.Fprintf(stdout, "Commit:         %s\n", result.Commit)
	}
	if result.CommitURL != "" {
		fmt.Fprintf(stdout, "Commit URL:     %s\n", result.CommitURL)
	}
	if result.PRURL != "" {
		fmt.Fprintf(stdout, "Pull request:   %s\n", result.PRURL)
	}
	return 0
}

func shareUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego share manifest <model-dir> --out <file> [flags]")
	fmt.Fprintln(w, "  nego share package <model-dir> --out <archive.tar.gz> [flags]")
	fmt.Fprintln(w, "  nego share upload <repo-id> <local-path> [path-in-repo] [flags]")
}

func runTrain(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "init":
			return runTrainInit(args[1:], stdout, stderr)
		case "validate":
			return runTrainValidate(args[1:], stdout, stderr)
		case "check":
			return runTrainCheck(args[1:], stdout, stderr)
		case "native":
			return runTrainNative(args[1:], stdout, stderr)
		case "help", "-h", "--help":
			trainUsage(stdout)
			return 0
		}
	}
	var jsonOutput bool
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego train <job.json> [flags]")
		return 2
	}
	data, err := os.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	var spec training.JobSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if !jsonOutput {
		fmt.Fprintln(stdout, "Checking training job...")
		report, err := training.Preflight(spec)
		if err != nil {
			fmt.Fprintf(stderr, "nego: %v\n", err)
			return 1
		}
		printTrainingPreflight(stdout, report)
		fmt.Fprintln(stdout, "Starting training process...")
	}
	result, err := training.Run(context.Background(), spec)
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(result)
	} else {
		status := "failed"
		if result.Success {
			status = "completed"
		}
		fmt.Fprintf(stdout, "Training job %s: %s (%s)\n", result.Name, status, result.Duration)
		if result.Stdout != "" {
			fmt.Fprint(stdout, result.Stdout)
		}
		if result.Stderr != "" {
			fmt.Fprint(stderr, result.Stderr)
		}
	}
	if err != nil {
		return 1
	}
	return 0
}

func runTrainInit(args []string, stdout, stderr io.Writer) int {
	var name, method, baseModel, trainFile, evalFile, datasetFormat, outputDir, command, script, workDir, out string
	fs := flag.NewFlagSet("train init", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&name, "name", "qwen3-lora", "training job name")
	fs.StringVar(&method, "method", "lora", "training method")
	fs.StringVar(&baseModel, "base-model", "./models/qwen3", "downloaded Hugging Face model directory")
	fs.StringVar(&trainFile, "train-file", "", "training dataset JSONL file")
	fs.StringVar(&evalFile, "eval-file", "", "evaluation dataset JSONL file")
	fs.StringVar(&datasetFormat, "dataset-format", "auto", "dataset format: auto, chat, completion, or instruction")
	fs.StringVar(&outputDir, "output-dir", "./outputs/qwen3-lora", "trained model or adapter output directory")
	fs.StringVar(&command, "command", "python", "training command")
	fs.StringVar(&script, "script", "scripts/train_lora.py", "training script passed as first argument")
	fs.StringVar(&workDir, "work-dir", ".", "training working directory")
	fs.StringVar(&out, "out", "train-job.json", "output job JSON file")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 0 || trainFile == "" {
		fmt.Fprintln(stderr, "usage: nego train init --base-model <dir> --train-file <file> --out <job.json> [flags]")
		return 2
	}
	spec, err := training.NewLoRAJob(training.InitOptions{
		Name:          name,
		Method:        method,
		BaseModel:     baseModel,
		TrainFile:     trainFile,
		EvalFile:      evalFile,
		DatasetFormat: datasetFormat,
		OutputDir:     outputDir,
		Command:       command,
		Script:        script,
		WorkDir:       workDir,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if err := training.WriteJob(out, spec); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Training job: %s\n", out)
	fmt.Fprintf(stdout, "Base model:   %s\n", spec.BaseModel)
	fmt.Fprintf(stdout, "Train file:   %s\n", spec.TrainFile)
	fmt.Fprintf(stdout, "Output dir:   %s\n", spec.OutputDir)
	return 0
}

func runTrainCheck(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("train check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON report")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego train check <job.json> [flags]")
		return 2
	}
	data, err := os.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	var spec training.JobSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	report, err := training.Preflight(spec)
	if jsonOutput {
		body := map[string]any{
			"valid":  err == nil,
			"report": report,
		}
		if err != nil {
			body["error"] = err.Error()
		}
		_ = json.NewEncoder(stdout).Encode(body)
	} else if err == nil {
		fmt.Fprintln(stdout, "Training job check passed")
		printTrainingPreflight(stdout, report)
	} else {
		fmt.Fprintf(stderr, "nego: %v\n", err)
	}
	if err != nil {
		return 1
	}
	return 0
}

func runTrainValidate(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("train validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego train validate <job.json> [flags]")
		return 2
	}
	data, err := os.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	var spec training.JobSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	err = training.Validate(spec)
	result := map[string]any{
		"name":  spec.Name,
		"valid": err == nil,
	}
	if err != nil {
		result["error"] = err.Error()
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(result)
	} else if err == nil {
		name := spec.Name
		if name == "" {
			name = positionals[0]
		}
		fmt.Fprintf(stdout, "Training job is valid: %s\n", name)
	} else {
		fmt.Fprintf(stderr, "nego: %v\n", err)
	}
	if err != nil {
		return 1
	}
	return 0
}

func runTrainNative(args []string, stdout, stderr io.Writer) int {
	var trainFile string
	var evalFile string
	var datasetFormat string
	var outputDir string
	var method string
	var learningRate float64
	var epochs int
	var jsonOutput bool
	fs := flag.NewFlagSet("train native", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&trainFile, "train-file", "", "training dataset file")
	fs.StringVar(&evalFile, "eval-file", "", "evaluation dataset file")
	fs.StringVar(&datasetFormat, "dataset-format", "auto", "dataset format: auto, chat, completion, or instruction")
	fs.StringVar(&outputDir, "out", "", "output adapter directory")
	fs.StringVar(&method, "method", "token-bias", "native training method")
	fs.Float64Var(&learningRate, "learning-rate", 0.1, "adapter learning-rate scale")
	fs.IntVar(&epochs, "epochs", 1, "number of passes over the dataset")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || trainFile == "" || outputDir == "" {
		fmt.Fprintln(stderr, "usage: nego train native <model-path> --train-file <file> --out <dir> [flags]")
		return 2
	}
	result, err := training.RunNative(context.Background(), training.NativeOptions{
		BaseModel:     positionals[0],
		TrainFile:     trainFile,
		EvalFile:      evalFile,
		DatasetFormat: datasetFormat,
		OutputDir:     outputDir,
		Method:        method,
		LearningRate:  learningRate,
		Epochs:        epochs,
	})
	if jsonOutput {
		body := map[string]any{
			"success": err == nil,
			"result":  result,
		}
		if err != nil {
			body["error"] = err.Error()
		}
		_ = json.NewEncoder(stdout).Encode(body)
	} else if err == nil {
		printNativeTrainingResult(stdout, result)
	} else {
		fmt.Fprintf(stderr, "nego: %v\n", err)
	}
	if err != nil {
		return 1
	}
	return 0
}

func printNativeTrainingResult(w io.Writer, result training.NativeResult) {
	fmt.Fprintln(w, "Native training completed")
	fmt.Fprintf(w, "Base model:     %s\n", result.BaseModel)
	fmt.Fprintf(w, "Train file:     %s (%d rows)\n", result.TrainFile, result.TrainRows)
	if result.EvalFile != "" {
		fmt.Fprintf(w, "Eval file:      %s (%d rows)\n", result.EvalFile, result.EvalRows)
	}
	fmt.Fprintf(w, "Dataset format: %s\n", result.DatasetFormat)
	fmt.Fprintf(w, "Method:         %s\n", result.Method)
	fmt.Fprintf(w, "Epochs:         %d\n", result.Epochs)
	fmt.Fprintf(w, "Train tokens:   %d\n", result.TrainTokens)
	fmt.Fprintf(w, "Updated tokens: %d\n", result.UpdatedTokens)
	fmt.Fprintf(w, "Adapter:        %s\n", result.AdapterPath)
	for _, warning := range result.Warnings {
		fmt.Fprintf(w, "Warning:        %s\n", warning)
	}
}

func printTrainingPreflight(w io.Writer, report training.PreflightReport) {
	name := report.Name
	if name == "" {
		name = "(unnamed)"
	}
	fmt.Fprintf(w, "Job:            %s\n", name)
	if report.Method != "" {
		fmt.Fprintf(w, "Method:         %s\n", report.Method)
	}
	if report.BaseModel != "" {
		fmt.Fprintf(w, "Base model:     %s\n", report.BaseModel)
	}
	if report.ModelType != "" {
		fmt.Fprintf(w, "Model type:     %s\n", report.ModelType)
	}
	if len(report.Architectures) > 0 {
		fmt.Fprintf(w, "Architecture:   %s\n", strings.Join(report.Architectures, ", "))
	}
	if report.TrainFile != "" {
		fmt.Fprintf(w, "Train file:     %s (%d rows)\n", report.TrainFile, report.TrainRows)
	}
	if report.EvalFile != "" {
		fmt.Fprintf(w, "Eval file:      %s (%d rows)\n", report.EvalFile, report.EvalRows)
	}
	if report.DatasetFormat != "" {
		fmt.Fprintf(w, "Dataset format: %s\n", report.DatasetFormat)
	}
	if report.OutputDir != "" {
		fmt.Fprintf(w, "Output dir:     %s\n", report.OutputDir)
	}
	if report.Command != "" {
		command := strings.TrimSpace(strings.Join(append([]string{report.Command}, redactTrainingArgs(report.Args)...), " "))
		fmt.Fprintf(w, "Command:        %s\n", command)
	}
	for _, warning := range report.Warnings {
		fmt.Fprintf(w, "Warning:        %s\n", warning)
	}
}

func redactTrainingArgs(args []string) []string {
	out := append([]string(nil), args...)
	redactNext := false
	for i, arg := range out {
		if redactNext {
			out[i] = "<redacted>"
			redactNext = false
			continue
		}
		lower := strings.ToLower(arg)
		if key, _, ok := strings.Cut(lower, "="); ok && sensitiveTrainingArg(key) {
			out[i] = arg[:strings.Index(arg, "=")+1] + "<redacted>"
			continue
		}
		trimmed := strings.TrimLeft(lower, "-")
		if sensitiveTrainingArg(trimmed) {
			redactNext = true
		}
	}
	return out
}

func sensitiveTrainingArg(key string) bool {
	key = strings.TrimSpace(strings.TrimLeft(strings.ToLower(key), "-"))
	return strings.Contains(key, "token") ||
		strings.Contains(key, "api-key") ||
		strings.Contains(key, "api_key") ||
		strings.Contains(key, "apikey") ||
		strings.Contains(key, "access-key") ||
		strings.Contains(key, "access_key") ||
		strings.Contains(key, "secret") ||
		strings.Contains(key, "password")
}

func trainUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego train init --base-model <dir> --train-file <file> --out <job.json> [flags]")
	fmt.Fprintln(w, "  nego train check <job.json> [flags]")
	fmt.Fprintln(w, "  nego train validate <job.json> [flags]")
	fmt.Fprintln(w, "  nego train native <model-path> --train-file <file> --out <dir> [flags]")
	fmt.Fprintln(w, "  nego train <job.json> [flags]")
}

func runConvert(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		convertUsage(stderr)
		return 2
	}
	switch args[0] {
	case "gguf":
		return runConvertGGUF(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		convertUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown convert command %q\n", args[0])
		convertUsage(stderr)
		return 2
	}
}

func runConvertGGUF(args []string, stdout, stderr io.Writer) int {
	var output string
	var converter string
	var python string
	var quantize string
	fs := flag.NewFlagSet("convert gguf", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&output, "out", "", "output GGUF path")
	fs.StringVar(&converter, "converter", "", "path to llama.cpp conversion script/binary")
	fs.StringVar(&python, "python", "", "python executable for converter scripts")
	fs.StringVar(&quantize, "quantize", "", "output quantization type")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego convert gguf <model-dir> --out <file> [--converter <path>] [--python <path>]")
		return 2
	}
	if err := convert.ConvertGGUF(context.Background(), convert.GGUFOptions{
		Converter: converter,
		Python:    python,
		ModelDir:  positionals[0],
		Output:    output,
		Quantize:  quantize,
	}); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Converted GGUF: %s\n", output)
	return 0
}

func convertUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego convert gguf <model-dir> --out <file> [--converter <path>] [--python <path>]")
}

func runVersion(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("version", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	info := version.Info()
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(info)
		return 0
	}
	fmt.Fprintf(stdout, "nego %s\ncommit %s\nbuilt %s\n", info["version"], info["commit"], info["date"])
	return 0
}

type evalConfig struct {
	Backend  string       `json:"backend"`
	Path     string       `json:"path"`
	Endpoint string       `json:"endpoint"`
	Model    string       `json:"model"`
	APIKey   string       `json:"api_key"`
	Cases    []evals.Case `json:"cases"`
}

const maxEvalReportBytes = 64 << 20
const maxChatSessionBytes = 16 << 20

func runEval(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "report":
			return runEvalReport(args[1:], stdout, stderr)
		case "compare":
			return runEvalCompare(args[1:], stdout, stderr)
		case "help", "-h", "--help":
			evalUsage(stdout)
			return 0
		}
	}
	var jsonOutput bool
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego eval <suite.json> [flags]")
		return 2
	}
	data, err := os.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	var cfg evalConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if cfg.Backend == "" {
		cfg.Backend = "openai-compatible"
	}
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Backend:  cfg.Backend,
		Path:     cfg.Path,
		Endpoint: cfg.Endpoint,
		Model:    cfg.Model,
		APIKey:   cfg.APIKey,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	defer model.Close()
	report := evals.Run(context.Background(), model, evals.Suite{Cases: cfg.Cases})
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(report)
	} else {
		fmt.Fprintf(stdout, "Passed: %d\nFailed: %d\n", report.Passed, report.Failed)
		for _, result := range report.Results {
			status := "FAIL"
			if result.Passed {
				status = "PASS"
			}
			fmt.Fprintf(stdout, "%s %s (%s)\n", status, result.Name, result.Duration)
			if result.Error != "" {
				fmt.Fprintf(stdout, "  %s\n", result.Error)
			}
		}
	}
	if report.Failed > 0 {
		return 1
	}
	return 0
}

func runEvalReport(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("eval report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego eval report <report.json> [flags]")
		return 2
	}
	report, err := readEvalReport(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	summary := evals.Summarize(report)
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(summary)
		return 0
	}
	writeEvalSummary(stdout, summary)
	return 0
}

func runEvalCompare(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("eval compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 2 {
		fmt.Fprintln(stderr, "usage: nego eval compare <baseline.json> <candidate.json> [flags]")
		return 2
	}
	baseline, err := readEvalReport(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	candidate, err := readEvalReport(positionals[1])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	comparison := evals.Compare(baseline, candidate)
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(comparison)
		return 0
	}
	fmt.Fprintln(stdout, "Baseline:")
	writeEvalSummary(stdout, comparison.Baseline)
	fmt.Fprintln(stdout, "Candidate:")
	writeEvalSummary(stdout, comparison.Candidate)
	if len(comparison.Improvements) > 0 {
		fmt.Fprintln(stdout, "Improvements:")
		for _, delta := range comparison.Improvements {
			fmt.Fprintf(stdout, "  PASS %s\n", delta.Name)
		}
	}
	if len(comparison.Regressions) > 0 {
		fmt.Fprintln(stdout, "Regressions:")
		for _, delta := range comparison.Regressions {
			fmt.Fprintf(stdout, "  FAIL %s", delta.Name)
			if delta.CandidateError != "" {
				fmt.Fprintf(stdout, " - %s", delta.CandidateError)
			}
			fmt.Fprintln(stdout)
		}
	}
	if len(comparison.Regressions) > 0 {
		return 1
	}
	return 0
}

func readEvalReport(path string) (evals.Report, error) {
	file, err := os.Open(path)
	if err != nil {
		return evals.Report{}, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxEvalReportBytes+1))
	if err != nil {
		return evals.Report{}, err
	}
	if len(data) > maxEvalReportBytes {
		return evals.Report{}, fmt.Errorf("eval report exceeds %s", humanBytes(maxEvalReportBytes))
	}
	var report evals.Report
	if err := json.Unmarshal(data, &report); err != nil {
		return evals.Report{}, err
	}
	return report, nil
}

func writeEvalSummary(w io.Writer, summary evals.Summary) {
	fmt.Fprintf(w, "Passed:   %d\n", summary.Passed)
	fmt.Fprintf(w, "Failed:   %d\n", summary.Failed)
	fmt.Fprintf(w, "Total:    %d\n", summary.Total)
	fmt.Fprintf(w, "PassRate: %.2f%%\n", summary.PassRate*100)
	fmt.Fprintf(w, "Duration: %s\n", summary.Duration)
	if len(summary.FailedCases) > 0 {
		fmt.Fprintln(w, "Failed cases:")
		for _, name := range summary.FailedCases {
			fmt.Fprintf(w, "  - %s\n", name)
		}
	}
}

func evalUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego eval <suite.json> [flags]")
	fmt.Fprintln(w, "  nego eval report <report.json> [flags]")
	fmt.Fprintln(w, "  nego eval compare <baseline.json> <candidate.json> [flags]")
}

func runTokenize(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("tokenize", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) < 2 {
		fmt.Fprintln(stderr, "usage: nego tokenize <model-path> <text> [flags]")
		return 2
	}
	tok, err := tokenizer.Load(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	ids, err := tok.Encode(strings.Join(positionals[1:], " "))
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(map[string]any{"tokens": ids, "count": len(ids)})
		return 0
	}
	for i, id := range ids {
		if i > 0 {
			fmt.Fprint(stdout, " ")
		}
		fmt.Fprint(stdout, id)
	}
	fmt.Fprintln(stdout)
	return 0
}

func runTokens(args []string, stdout, stderr io.Writer) int {
	if len(args) < 2 {
		fmt.Fprintln(stderr, "usage: nego tokens <model-path> <text>")
		return 2
	}
	tok, err := tokenizer.Load(args[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	count, err := tok.Count(strings.Join(args[1:], " "))
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, count)
	return 0
}

func runContext(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	var maxContext int
	var text string
	var system string
	var users repeatedFlag
	var assistants repeatedFlag

	fs := flag.NewFlagSet("context", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	fs.IntVar(&maxContext, "max-context", 0, "maximum context tokens")
	fs.StringVar(&text, "text", "", "raw text to count")
	fs.StringVar(&system, "system", "", "system message")
	fs.Var(&users, "user", "user message, repeatable")
	fs.Var(&assistants, "assistant", "assistant message, repeatable")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) < 1 {
		fmt.Fprintln(stderr, "usage: nego context <model-path> <text> [flags]")
		return 2
	}
	modelPath := positionals[0]
	if text == "" && len(positionals) > 1 {
		text = strings.Join(positionals[1:], " ")
	}
	prompt := text
	messages := buildPromptMessages(system, users, assistants)
	if prompt == "" && len(messages) > 0 {
		rendered, err := chattemplate.Render(modelPath, messages, chattemplate.Options{AddGenerationPrompt: true})
		if err != nil {
			fmt.Fprintf(stderr, "nego: %v\n", err)
			return 1
		}
		prompt = rendered
	}
	if prompt == "" {
		fmt.Fprintln(stderr, "usage: nego context <model-path> <text> [flags]")
		return 2
	}
	tok, err := tokenizer.Load(modelPath)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	tokenCount, err := tok.Count(prompt)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if maxContext == 0 {
		if report, err := modelinfo.Check(modelPath); err == nil && report.ContextLength > 0 {
			maxContext = int(report.ContextLength)
		}
	}
	result := contextBudgetResult{
		Tokens:     tokenCount,
		MaxContext: maxContext,
		Fits:       maxContext == 0 || tokenCount <= maxContext,
	}
	if maxContext > 0 {
		result.Remaining = maxContext - tokenCount
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(result)
		return 0
	}
	fmt.Fprintf(stdout, "Tokens:      %d\n", result.Tokens)
	if result.MaxContext > 0 {
		fmt.Fprintf(stdout, "Max context: %d\n", result.MaxContext)
		fmt.Fprintf(stdout, "Remaining:   %d\n", result.Remaining)
	}
	if result.Fits {
		fmt.Fprintln(stdout, "Fits:        yes")
		return 0
	}
	fmt.Fprintln(stdout, "Fits:        no")
	return 1
}

type contextBudgetResult struct {
	Tokens     int  `json:"tokens"`
	MaxContext int  `json:"max_context,omitempty"`
	Remaining  int  `json:"remaining,omitempty"`
	Fits       bool `json:"fits"`
}

func buildPromptMessages(system string, users, assistants []string) []chattemplate.Message {
	var messages []chattemplate.Message
	if system != "" {
		messages = append(messages, chattemplate.Message{Role: chattemplate.RoleSystem, Content: system})
	}
	maxLen := max(len(users), len(assistants))
	for i := 0; i < maxLen; i++ {
		if i < len(users) {
			messages = append(messages, chattemplate.Message{Role: chattemplate.RoleUser, Content: users[i]})
		}
		if i < len(assistants) {
			messages = append(messages, chattemplate.Message{Role: chattemplate.RoleAssistant, Content: assistants[i]})
		}
	}
	return messages
}

func runPrompt(args []string, stdout, stderr io.Writer) int {
	var system string
	var users repeatedFlag
	var assistants repeatedFlag
	var noGenerationPrompt bool

	fs := flag.NewFlagSet("prompt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&system, "system", "", "system message")
	fs.Var(&users, "user", "user message, repeatable")
	fs.Var(&assistants, "assistant", "assistant message, repeatable")
	fs.BoolVar(&noGenerationPrompt, "no-generation-prompt", false, "do not append assistant generation prompt")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || (system == "" && len(users) == 0 && len(assistants) == 0) {
		fmt.Fprintln(stderr, "usage: nego prompt <model-path> --user <text> [flags]")
		return 2
	}
	messages := buildPromptMessages(system, users, assistants)
	prompt, err := chattemplate.Render(positionals[0], messages, chattemplate.Options{AddGenerationPrompt: !noGenerationPrompt})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, prompt)
	return 0
}

func runInspect(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego inspect <model-path> [flags]")
		return 2
	}
	info, err := modelinfo.Inspect(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(info)
		return 0
	}
	writeInspectInfo(stdout, info)
	return 0
}

func writeInspectInfo(w io.Writer, info *modelinfo.Info) {
	fmt.Fprintf(w, "Path:        %s\n", info.Path)
	if info.ModelType != "" {
		fmt.Fprintf(w, "Model type:  %s\n", info.ModelType)
	}
	if len(info.Architectures) > 0 {
		fmt.Fprintf(w, "Architecture:%s\n", " "+strings.Join(info.Architectures, ", "))
	}
	if info.GGUF != nil {
		fmt.Fprintln(w, "GGUF:")
		fmt.Fprintf(w, "  Version:       %d\n", info.GGUF.Version)
		fmt.Fprintf(w, "  Tensors:       %d\n", info.GGUF.TensorCount)
		fmt.Fprintf(w, "  Metadata:      %d\n", info.GGUF.MetadataCount)
		if info.GGUF.Quantization != "" {
			fmt.Fprintf(w, "  Quantization:  %s\n", info.GGUF.Quantization)
		}
		if info.GGUF.ContextLength > 0 {
			fmt.Fprintf(w, "  Context:       %d\n", info.GGUF.ContextLength)
		}
		if info.GGUF.EmbeddingLength > 0 {
			fmt.Fprintf(w, "  Embedding:     %d\n", info.GGUF.EmbeddingLength)
		}
		if info.GGUF.BlockCount > 0 {
			fmt.Fprintf(w, "  Blocks:        %d\n", info.GGUF.BlockCount)
		}
		if info.GGUF.VocabSize > 0 {
			fmt.Fprintf(w, "  Vocab:         %d\n", info.GGUF.VocabSize)
		}
		if info.GGUF.ChatTemplate != "" {
			fmt.Fprintln(w, "  Chat template: yes")
		}
	}
	if info.Safetensors != nil {
		fmt.Fprintln(w, "Safetensors:")
		fmt.Fprintf(w, "  Files:        %d\n", len(info.Safetensors.Files))
		fmt.Fprintf(w, "  Tensors:      %d\n", len(info.Safetensors.Tensors))
		if info.Safetensors.ParamCount > 0 {
			fmt.Fprintf(w, "  Parameters:   %d\n", info.Safetensors.ParamCount)
		}
		if info.Safetensors.TotalSize > 0 {
			fmt.Fprintf(w, "  Tensor bytes: %s\n", humanBytes(int64(info.Safetensors.TotalSize)))
		}
		if len(info.Safetensors.DTypeCounts) > 0 {
			fmt.Fprintf(w, "  DTypes:       %s\n", formatDTypeCounts(info.Safetensors.DTypeCounts))
		}
	}
	if info.HFWeights != nil {
		fmt.Fprintln(w, "HF weights:")
		fmt.Fprintf(w, "  Ready:        %v\n", info.HFWeights.Ready)
		if info.HFWeights.BlockCount > 0 {
			fmt.Fprintf(w, "  Blocks:       %d\n", info.HFWeights.BlockCount)
		}
		if info.HFWeights.TokenEmbedding != "" {
			fmt.Fprintf(w, "  Embedding:    %s\n", info.HFWeights.TokenEmbedding)
		}
		if info.HFWeights.TiedOutput {
			fmt.Fprintln(w, "  Output:       tied")
		} else if info.HFWeights.Output != "" {
			fmt.Fprintf(w, "  Output:       %s\n", info.HFWeights.Output)
		}
		if len(info.HFWeights.Missing) > 0 {
			fmt.Fprintf(w, "  Missing:      %d\n", len(info.HFWeights.Missing))
		}
	}
	if info.Card != nil {
		fmt.Fprintln(w, "Model card:")
		if info.Card.Title != "" {
			fmt.Fprintf(w, "  Title:        %s\n", info.Card.Title)
		}
		if info.Card.License != "" {
			fmt.Fprintf(w, "  License:      %s\n", info.Card.License)
		}
		if info.Card.PipelineTag != "" {
			fmt.Fprintf(w, "  Pipeline:     %s\n", info.Card.PipelineTag)
		}
		if info.Card.LibraryName != "" {
			fmt.Fprintf(w, "  Library:      %s\n", info.Card.LibraryName)
		}
		if len(info.Card.Tags) > 0 {
			fmt.Fprintf(w, "  Tags:         %s\n", strings.Join(info.Card.Tags, ", "))
		}
		if len(info.Card.Languages) > 0 {
			fmt.Fprintf(w, "  Languages:    %s\n", strings.Join(info.Card.Languages, ", "))
		}
		if len(info.Card.Datasets) > 0 {
			fmt.Fprintf(w, "  Datasets:     %s\n", strings.Join(info.Card.Datasets, ", "))
		}
		if len(info.Card.BaseModels) > 0 {
			fmt.Fprintf(w, "  Base models:  %s\n", strings.Join(info.Card.BaseModels, ", "))
		}
	}
	if len(info.Files) > 0 {
		var total int64
		for _, file := range info.Files {
			total += file.Size
		}
		fmt.Fprintf(w, "Files:       %d (%s)\n", len(info.Files), humanBytes(total))
		for _, file := range info.Files {
			fmt.Fprintf(w, "  - %s", file.Path)
			if file.Kind != "" {
				fmt.Fprintf(w, " [%s]", file.Kind)
			}
			if file.Size > 0 {
				fmt.Fprintf(w, " %s", humanBytes(file.Size))
			}
			fmt.Fprintln(w)
		}
	}
}

func formatDTypeCounts(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", key, counts[key]))
	}
	return strings.Join(parts, ", ")
}

func runCheck(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego check <model-path> [flags]")
		return 2
	}
	report, err := modelinfo.Check(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(report)
		return 0
	}
	writeCheckReport(stdout, report)
	return 0
}

func writeCheckReport(w io.Writer, report *modelinfo.CheckReport) {
	fmt.Fprintf(w, "Path:          %s\n", report.Path)
	if report.ModelType != "" {
		fmt.Fprintf(w, "Model type:    %s\n", report.ModelType)
	}
	if report.Architecture != "" {
		fmt.Fprintf(w, "Architecture:  %s\n", report.Architecture)
	}
	if report.RuntimeFile != nil {
		fmt.Fprintf(w, "Runtime file:  %s [%s]\n", report.RuntimeFile.Path, report.RuntimeFile.Kind)
	}
	if report.ContextLength > 0 {
		fmt.Fprintf(w, "Context:       %d\n", report.ContextLength)
	}
	if report.Quantization != "" {
		fmt.Fprintf(w, "Quantization:  %s\n", report.Quantization)
	}
	if report.ChatTemplate {
		fmt.Fprintln(w, "Chat template: yes")
	} else {
		fmt.Fprintln(w, "Chat template: no")
	}
	if report.Artifact != nil {
		fmt.Fprintln(w, "Artifact:")
		fmt.Fprintf(w, "  Format:       %s\n", report.Artifact.Format)
		if report.Artifact.RecommendedRunBackend != "" {
			fmt.Fprintf(w, "  Run:          %s\n", report.Artifact.RecommendedRunBackend)
		} else {
			fmt.Fprintln(w, "  Run:          not ready")
		}
		if report.Artifact.RecommendedTrainBackend != "" {
			fmt.Fprintf(w, "  Train:        %s\n", report.Artifact.RecommendedTrainBackend)
		} else {
			fmt.Fprintln(w, "  Train:        not ready")
		}
		if report.Artifact.ParameterCount > 0 {
			fmt.Fprintf(w, "  Parameters:   %d\n", report.Artifact.ParameterCount)
		}
	}
	fmt.Fprintln(w, "Backends:")
	for _, backend := range report.Backends {
		status := "no"
		if backend.Compatible {
			status = "yes"
		}
		fmt.Fprintf(w, "  - %s: %s", backend.Name, status)
		if backend.Reason != "" {
			fmt.Fprintf(w, " (%s)", backend.Reason)
		}
		fmt.Fprintln(w)
	}
	if len(report.Warnings) > 0 {
		fmt.Fprintln(w, "Warnings:")
		for _, warning := range report.Warnings {
			fmt.Fprintf(w, "  - %s\n", warning)
		}
	}
}

func runModel(args []string, stdout, stderr io.Writer) int {
	var backend string
	var maxTokens int
	var temperature float64
	var topP float64
	var repeatPenalty float64
	var seed int64
	var stop repeatedFlag
	var threads int
	var ctxSize int
	var gpuLayers int
	var gpuMode string
	var mainGPU int
	var tensorSplit string
	var splitMode string
	var flashAttention bool
	var native bool
	var adapterPath string
	var configFile string
	var logPath string
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "auto", "runtime backend: auto, native, llama.cpp, or openai-compatible")
	fs.BoolVar(&native, "native", false, "use the experimental pure-Go native backend")
	fs.IntVar(&maxTokens, "max-tokens", 0, "maximum tokens to generate")
	fs.Float64Var(&temperature, "temperature", 0, "sampling temperature")
	fs.Float64Var(&topP, "top-p", 0, "nucleus sampling probability")
	fs.Float64Var(&repeatPenalty, "repeat-penalty", 0, "penalty for repeated tokens, >= 1")
	fs.Int64Var(&seed, "seed", 0, "random seed")
	fs.Var(&stop, "stop", "stop sequence, repeatable")
	fs.IntVar(&threads, "threads", 0, "llama.cpp CPU threads")
	fs.IntVar(&ctxSize, "ctx-size", 0, "llama.cpp context size")
	fs.IntVar(&gpuLayers, "gpu-layers", 0, "llama.cpp GPU layers")
	fs.StringVar(&gpuMode, "gpu", "", "llama.cpp GPU mode: off, auto, or full")
	fs.IntVar(&mainGPU, "main-gpu", -1, "llama.cpp main GPU index")
	fs.StringVar(&tensorSplit, "tensor-split", "", "llama.cpp comma-separated tensor split")
	fs.StringVar(&splitMode, "split-mode", "", "llama.cpp multi-GPU split mode")
	fs.BoolVar(&flashAttention, "flash-attn", false, "enable llama.cpp flash attention")
	fs.StringVar(&adapterPath, "adapter", "", "native adapter JSON")
	fs.StringVar(&configFile, "f", "", "run config file")
	fs.StringVar(&logPath, "log", "", "append run result to JSONL log")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	backendSetByFlag := flagWasSet(fs, "backend")
	if native && backendSetByFlag {
		fmt.Fprintln(stderr, "nego: use either --native or --backend, not both")
		return 2
	}
	cfg, err := loadRuntimeConfig(configFile)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if cfg.Backend != "" {
		backend = cfg.Backend
	}
	if native {
		backend = "native"
	}
	backend = normalizeBackendFlag(backend)
	if cfg.MaxTokens > 0 && maxTokens == 0 {
		maxTokens = cfg.MaxTokens
	}
	if cfg.Temperature > 0 && temperature == 0 {
		temperature = cfg.Temperature
	}
	if cfg.TopP > 0 && topP == 0 {
		topP = cfg.TopP
	}
	if cfg.RepeatPenalty > 0 && repeatPenalty == 0 {
		repeatPenalty = cfg.RepeatPenalty
	}
	if err := validateRepeatPenalty(repeatPenalty); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if cfg.Seed != 0 && seed == 0 {
		seed = cfg.Seed
	}
	if len(stop) == 0 {
		stop = append(stop, cfg.Stop...)
	}
	if cfg.Log != "" && logPath == "" {
		logPath = cfg.Log
	}
	path := cfg.Path
	prompt := cfg.Prompt
	if len(positionals) > 0 {
		path = positionals[0]
	}
	if len(positionals) > 1 {
		prompt = strings.Join(positionals[1:], " ")
	}
	if path == "" || prompt == "" {
		fmt.Fprintln(stderr, "usage: nego run <model-path> <prompt> [flags]")
		return 2
	}
	options := runtimeOptions(cfg.Options, runtimeFlagOptions{
		threads:        threads,
		ctxSize:        ctxSize,
		gpuLayers:      gpuLayers,
		gpuMode:        gpuMode,
		mainGPU:        mainGPU,
		tensorSplit:    tensorSplit,
		splitMode:      splitMode,
		flashAttention: flashAttention,
		adapterPath:    adapterPath,
	})
	if err := validateRuntimeOptions(options); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	started := time.Now().UTC()
	backend = resolveRuntimeBackend(backend, path, cfg.Endpoint, cfg.Model)
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{Backend: backend, Path: path, Endpoint: cfg.Endpoint, Model: cfg.Model, APIKey: cfg.APIKey, Options: options})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("run", backend, path, cfg.Endpoint, cfg.Model, prompt, nil, "", started, maxTokens, temperature, topP, repeatPenalty, stop, seed, options, err)); logErr != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
		}
		return 1
	}
	defer model.Close()
	out, err := model.Generate(context.Background(), nego.GenerateRequest{Prompt: prompt, MaxTokens: maxTokens, Temperature: temperature, TopP: topP, RepeatPenalty: repeatPenalty, Stop: stop, Seed: seed})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("run", backend, path, cfg.Endpoint, cfg.Model, prompt, nil, "", started, maxTokens, temperature, topP, repeatPenalty, stop, seed, options, err)); logErr != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
		}
		return 1
	}
	fmt.Fprint(stdout, out.Text)
	if err := appendRuntimeLog(logPath, runtimeLogEntry("run", backend, path, cfg.Endpoint, cfg.Model, prompt, nil, out.Text, started, maxTokens, temperature, topP, repeatPenalty, stop, seed, options, nil)); err != nil {
		fmt.Fprintf(stderr, "nego: write run log: %v\n", err)
		return 1
	}
	return 0
}

func runChat(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	var backend string
	var system string
	var maxTokens int
	var temperature float64
	var topP float64
	var repeatPenalty float64
	var seed int64
	var stop repeatedFlag
	var threads int
	var ctxSize int
	var gpuLayers int
	var gpuMode string
	var mainGPU int
	var tensorSplit string
	var splitMode string
	var flashAttention bool
	var configFile string
	var logPath string
	var sessionPath string
	var savePath string
	var interactive bool
	var native bool
	var adapterPath string
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "auto", "runtime backend: auto, native, llama.cpp, or openai-compatible")
	fs.BoolVar(&native, "native", false, "use the experimental pure-Go native backend")
	fs.StringVar(&system, "system", "", "system message")
	fs.IntVar(&maxTokens, "max-tokens", 0, "maximum tokens to generate")
	fs.Float64Var(&temperature, "temperature", 0, "sampling temperature")
	fs.Float64Var(&topP, "top-p", 0, "nucleus sampling probability")
	fs.Float64Var(&repeatPenalty, "repeat-penalty", 0, "penalty for repeated tokens, >= 1")
	fs.Int64Var(&seed, "seed", 0, "random seed")
	fs.Var(&stop, "stop", "stop sequence, repeatable")
	fs.IntVar(&threads, "threads", 0, "llama.cpp CPU threads")
	fs.IntVar(&ctxSize, "ctx-size", 0, "llama.cpp context size")
	fs.IntVar(&gpuLayers, "gpu-layers", 0, "llama.cpp GPU layers")
	fs.StringVar(&gpuMode, "gpu", "", "llama.cpp GPU mode: off, auto, or full")
	fs.IntVar(&mainGPU, "main-gpu", -1, "llama.cpp main GPU index")
	fs.StringVar(&tensorSplit, "tensor-split", "", "llama.cpp comma-separated tensor split")
	fs.StringVar(&splitMode, "split-mode", "", "llama.cpp multi-GPU split mode")
	fs.BoolVar(&flashAttention, "flash-attn", false, "enable llama.cpp flash attention")
	fs.StringVar(&adapterPath, "adapter", "", "native adapter JSON")
	fs.StringVar(&configFile, "f", "", "chat config file")
	fs.StringVar(&logPath, "log", "", "append run result to JSONL log")
	fs.StringVar(&sessionPath, "session", "", "load and save chat history JSON")
	fs.StringVar(&savePath, "save", "", "save chat history JSON without loading it")
	fs.BoolVar(&interactive, "interactive", false, "start an interactive chat session")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	backendSetByFlag := flagWasSet(fs, "backend")
	if native && backendSetByFlag {
		fmt.Fprintln(stderr, "nego: use either --native or --backend, not both")
		return 2
	}
	cfg, err := loadRuntimeConfig(configFile)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if cfg.Backend != "" {
		backend = cfg.Backend
	}
	backend = normalizeBackendFlag(backend)
	if cfg.System != "" && system == "" {
		system = cfg.System
	}
	if cfg.MaxTokens > 0 && maxTokens == 0 {
		maxTokens = cfg.MaxTokens
	}
	if cfg.Temperature > 0 && temperature == 0 {
		temperature = cfg.Temperature
	}
	if cfg.TopP > 0 && topP == 0 {
		topP = cfg.TopP
	}
	if cfg.RepeatPenalty > 0 && repeatPenalty == 0 {
		repeatPenalty = cfg.RepeatPenalty
	}
	if err := validateRepeatPenalty(repeatPenalty); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if cfg.Seed != 0 && seed == 0 {
		seed = cfg.Seed
	}
	if len(stop) == 0 {
		stop = append(stop, cfg.Stop...)
	}
	if cfg.Log != "" && logPath == "" {
		logPath = cfg.Log
	}
	if cfg.Session != "" && sessionPath == "" {
		sessionPath = cfg.Session
	}
	if cfg.Save != "" && savePath == "" {
		savePath = cfg.Save
	}
	if sessionPath != "" && savePath != "" {
		fmt.Fprintln(stderr, "nego: use either --session or --save, not both")
		return 2
	}
	path := cfg.Path
	if len(positionals) > 0 {
		path = positionals[0]
	}
	session, loadedSession, err := loadChatSession(sessionPath)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if loadedSession {
		if session.Backend != "" && cfg.Backend == "" && !backendSetByFlag {
			backend = session.Backend
		}
		if path == "" {
			path = session.Path
		}
		if cfg.Endpoint == "" {
			cfg.Endpoint = sanitizeEndpoint(session.Endpoint)
		}
		if cfg.Model == "" {
			cfg.Model = session.Model
		}
	}
	if native {
		backend = "native"
	}
	backend = normalizeBackendFlag(backend)
	messages := append([]nego.Message(nil), cfg.Messages...)
	if loadedSession {
		messages = append([]nego.Message(nil), session.Messages...)
	}
	if system != "" && len(messages) == 0 {
		messages = append(messages, nego.Message{Role: nego.RoleSystem, Content: system})
	}
	if len(positionals) > 1 {
		messages = append(messages, nego.Message{Role: nego.RoleUser, Content: strings.Join(positionals[1:], " ")})
	} else if cfg.Prompt != "" && len(messages) == 0 {
		messages = append(messages, nego.Message{Role: nego.RoleUser, Content: cfg.Prompt})
	}
	if path == "" || len(messages) == 0 {
		if !(interactive && path != "") {
			fmt.Fprintln(stderr, "usage: nego chat <model-path> <message> [flags]")
			return 2
		}
	}
	options := runtimeOptions(cfg.Options, runtimeFlagOptions{
		threads:        threads,
		ctxSize:        ctxSize,
		gpuLayers:      gpuLayers,
		gpuMode:        gpuMode,
		mainGPU:        mainGPU,
		tensorSplit:    tensorSplit,
		splitMode:      splitMode,
		flashAttention: flashAttention,
		adapterPath:    adapterPath,
	})
	if err := validateRuntimeOptions(options); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	backend = resolveRuntimeBackend(backend, path, cfg.Endpoint, cfg.Model)
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{Backend: backend, Path: path, Endpoint: cfg.Endpoint, Model: cfg.Model, APIKey: cfg.APIKey, Options: options})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		started := time.Now().UTC()
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("chat", backend, path, cfg.Endpoint, cfg.Model, "", messages, "", started, maxTokens, temperature, topP, repeatPenalty, stop, seed, options, err)); logErr != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
		}
		return 1
	}
	defer model.Close()
	if interactive {
		return runInteractiveChat(stdin, stdout, stderr, model, interactiveChatOptions{
			backend:       backend,
			path:          path,
			endpoint:      cfg.Endpoint,
			modelID:       cfg.Model,
			system:        system,
			messages:      messages,
			maxTokens:     maxTokens,
			temperature:   temperature,
			topP:          topP,
			repeatPenalty: repeatPenalty,
			stop:          stop,
			seed:          seed,
			options:       options,
			logPath:       logPath,
			sessionPath:   sessionSavePath(sessionPath, savePath),
			session:       session,
		})
	}
	started := time.Now().UTC()
	resp, err := model.Chat(context.Background(), nego.ChatRequest{Messages: messages, MaxTokens: maxTokens, Temperature: temperature, TopP: topP, RepeatPenalty: repeatPenalty, Stop: stop, Seed: seed})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("chat", backend, path, cfg.Endpoint, cfg.Model, "", messages, "", started, maxTokens, temperature, topP, repeatPenalty, stop, seed, options, err)); logErr != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
		}
		return 1
	}
	fmt.Fprint(stdout, resp.Message.Content)
	messages = append(messages, resp.Message)
	session = session.withMessages(backend, path, cfg.Endpoint, cfg.Model, messages)
	if err := saveChatSession(sessionSavePath(sessionPath, savePath), session); err != nil {
		fmt.Fprintf(stderr, "nego: save chat session: %v\n", err)
		return 1
	}
	if err := appendRuntimeLog(logPath, runtimeLogEntry("chat", backend, path, cfg.Endpoint, cfg.Model, "", messages, resp.Message.Content, started, maxTokens, temperature, topP, repeatPenalty, stop, seed, options, nil)); err != nil {
		fmt.Fprintf(stderr, "nego: write run log: %v\n", err)
		return 1
	}
	return 0
}

type interactiveChatOptions struct {
	backend       string
	path          string
	endpoint      string
	modelID       string
	system        string
	messages      []nego.Message
	maxTokens     int
	temperature   float64
	topP          float64
	repeatPenalty float64
	stop          []string
	seed          int64
	options       map[string]string
	logPath       string
	sessionPath   string
	session       chatSession
}

func runInteractiveChat(stdin io.Reader, stdout, stderr io.Writer, model nego.Model, opts interactiveChatOptions) int {
	messages := append([]nego.Message(nil), opts.messages...)
	if opts.system != "" && len(messages) == 0 {
		messages = append(messages, nego.Message{Role: nego.RoleSystem, Content: opts.system})
	}
	session := opts.session
	scanner := bufio.NewScanner(stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	fmt.Fprintln(stderr, "Interactive chat. Type /exit to quit, /reset to clear history.")
	for {
		fmt.Fprint(stderr, "user> ")
		if !scanner.Scan() {
			break
		}
		text := strings.TrimSpace(scanner.Text())
		switch text {
		case "":
			continue
		case "/exit", "/quit":
			return 0
		case "/reset":
			messages = nil
			if opts.system != "" {
				messages = append(messages, nego.Message{Role: nego.RoleSystem, Content: opts.system})
			}
			session.Messages = messages
			if err := saveChatSession(opts.sessionPath, session.withMessages(opts.backend, opts.path, opts.endpoint, opts.modelID, messages)); err != nil {
				fmt.Fprintf(stderr, "nego: save chat session: %v\n", err)
				return 1
			}
			fmt.Fprintln(stderr, "history reset")
			continue
		}
		messages = append(messages, nego.Message{Role: nego.RoleUser, Content: text})
		started := time.Now().UTC()
		fmt.Fprint(stdout, "assistant> ")
		stream, err := model.StreamChat(context.Background(), nego.ChatRequest{
			Messages:      messages,
			MaxTokens:     opts.maxTokens,
			Temperature:   opts.temperature,
			TopP:          opts.topP,
			RepeatPenalty: opts.repeatPenalty,
			Stop:          opts.stop,
			Seed:          opts.seed,
		})
		if err != nil {
			fmt.Fprintln(stdout)
			fmt.Fprintf(stderr, "nego: %v\n", err)
			if logErr := appendRuntimeLog(opts.logPath, runtimeLogEntry("chat", opts.backend, opts.path, opts.endpoint, opts.modelID, "", messages, "", started, opts.maxTokens, opts.temperature, opts.topP, opts.repeatPenalty, opts.stop, opts.seed, opts.options, err)); logErr != nil {
				fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
			}
			return 1
		}
		var output strings.Builder
		for token := range stream.Tokens() {
			fmt.Fprint(stdout, token.Text)
			output.WriteString(token.Text)
		}
		if err := stream.Err(); err != nil {
			_ = stream.Close()
			fmt.Fprintln(stdout)
			fmt.Fprintf(stderr, "nego: %v\n", err)
			if logErr := appendRuntimeLog(opts.logPath, runtimeLogEntry("chat", opts.backend, opts.path, opts.endpoint, opts.modelID, "", messages, output.String(), started, opts.maxTokens, opts.temperature, opts.topP, opts.repeatPenalty, opts.stop, opts.seed, opts.options, err)); logErr != nil {
				fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
			}
			return 1
		}
		_ = stream.Close()
		fmt.Fprintln(stdout)
		reply := output.String()
		messages = append(messages, nego.Message{Role: nego.RoleAssistant, Content: reply})
		session = session.withMessages(opts.backend, opts.path, opts.endpoint, opts.modelID, messages)
		if err := saveChatSession(opts.sessionPath, session); err != nil {
			fmt.Fprintf(stderr, "nego: save chat session: %v\n", err)
			return 1
		}
		if err := appendRuntimeLog(opts.logPath, runtimeLogEntry("chat", opts.backend, opts.path, opts.endpoint, opts.modelID, "", messages, reply, started, opts.maxTokens, opts.temperature, opts.topP, opts.repeatPenalty, opts.stop, opts.seed, opts.options, nil)); err != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", err)
			return 1
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	return 0
}

type runtimeConfig struct {
	Backend       string            `json:"backend"`
	Path          string            `json:"path"`
	Endpoint      string            `json:"endpoint"`
	Model         string            `json:"model"`
	APIKey        string            `json:"api_key"`
	Options       map[string]string `json:"options"`
	System        string            `json:"system"`
	Prompt        string            `json:"prompt"`
	Messages      []nego.Message    `json:"messages"`
	MaxTokens     int               `json:"max_tokens"`
	Temperature   float64           `json:"temperature"`
	TopP          float64           `json:"top_p"`
	RepeatPenalty float64           `json:"repeat_penalty"`
	Stop          []string          `json:"stop"`
	Seed          int64             `json:"seed"`
	Log           string            `json:"log"`
	Session       string            `json:"session"`
	Save          string            `json:"save"`
}

type chatSession struct {
	Backend  string         `json:"backend,omitempty"`
	Path     string         `json:"path,omitempty"`
	Endpoint string         `json:"endpoint,omitempty"`
	Model    string         `json:"model,omitempty"`
	Created  time.Time      `json:"created_at,omitempty"`
	Updated  time.Time      `json:"updated_at,omitempty"`
	Messages []nego.Message `json:"messages"`
}

func (s chatSession) withMessages(backend, path, endpoint, modelID string, messages []nego.Message) chatSession {
	s.Backend = backend
	s.Path = path
	s.Endpoint = sanitizeEndpoint(endpoint)
	s.Model = modelID
	s.Messages = append([]nego.Message(nil), messages...)
	return s
}

func sessionSavePath(sessionPath, savePath string) string {
	if sessionPath != "" {
		return sessionPath
	}
	return savePath
}

func loadChatSession(path string) (chatSession, bool, error) {
	if path == "" {
		return chatSession{}, false, nil
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return chatSession{}, false, nil
		}
		return chatSession{}, false, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxChatSessionBytes+1))
	if err != nil {
		return chatSession{}, false, err
	}
	if len(data) > maxChatSessionBytes {
		return chatSession{}, false, fmt.Errorf("chat session exceeds %s", humanBytes(maxChatSessionBytes))
	}
	var session chatSession
	if err := json.Unmarshal(data, &session); err != nil {
		return chatSession{}, false, err
	}
	return session, true, nil
}

func saveChatSession(path string, session chatSession) error {
	if path == "" {
		return nil
	}
	now := time.Now().UTC()
	if session.Created.IsZero() {
		session.Created = now
	}
	session.Updated = now
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(session)
}

func loadRuntimeConfig(path string) (runtimeConfig, error) {
	if path == "" {
		return runtimeConfig{}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return runtimeConfig{}, err
	}
	var cfg runtimeConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return runtimeConfig{}, err
	}
	return cfg, nil
}

func appendRuntimeLog(path string, entry runs.Entry) error {
	if path == "" {
		return nil
	}
	return runs.Append(path, entry)
}

func runtimeLogEntry(command, backend, path, endpoint, modelID, prompt string, messages []nego.Message, output string, started time.Time, maxTokens int, temperature, topP, repeatPenalty float64, stop []string, seed int64, options map[string]string, runErr error) runs.Entry {
	entry := runs.Entry{
		ID:            runs.NewID(),
		Command:       command,
		Backend:       backend,
		Path:          path,
		Model:         modelID,
		Endpoint:      sanitizeEndpoint(endpoint),
		Prompt:        prompt,
		Messages:      messages,
		Output:        output,
		StartedAt:     started,
		DurationMS:    time.Since(started).Milliseconds(),
		MaxTokens:     maxTokens,
		Temperature:   temperature,
		TopP:          topP,
		RepeatPenalty: repeatPenalty,
		Stop:          append([]string(nil), stop...),
		Seed:          seed,
		Options:       copyStringMap(options),
	}
	if runErr != nil {
		entry.Error = runErr.Error()
	}
	return entry
}

func sanitizeEndpoint(endpoint string) string {
	if endpoint == "" {
		return ""
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String()
}

func copyStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

type runtimeFlagOptions struct {
	threads        int
	ctxSize        int
	gpuLayers      int
	gpuMode        string
	mainGPU        int
	tensorSplit    string
	splitMode      string
	flashAttention bool
	adapterPath    string
}

func normalizeBackendFlag(backend string) string {
	if strings.EqualFold(strings.TrimSpace(backend), "auto") {
		return ""
	}
	return backend
}

func resolveRuntimeBackend(backend, path, endpoint, modelID string) string {
	backend = normalizeBackendFlag(backend)
	if backend != "" {
		return backend
	}
	resolved, err := nego.ResolveBackend(nego.ModelOptions{
		Path:     path,
		Endpoint: endpoint,
		Model:    modelID,
	})
	if err != nil {
		return ""
	}
	return resolved
}

func runtimeOptions(base map[string]string, flags runtimeFlagOptions) map[string]string {
	options := make(map[string]string, len(base)+9)
	for key, value := range base {
		if value != "" {
			options[key] = value
		}
	}
	if flags.threads > 0 {
		options["threads"] = strconv.Itoa(flags.threads)
	}
	if flags.ctxSize > 0 {
		options["ctx_size"] = strconv.Itoa(flags.ctxSize)
	}
	if flags.gpuLayers > 0 {
		options["gpu_layers"] = strconv.Itoa(flags.gpuLayers)
	}
	if flags.gpuMode != "" {
		options["gpu"] = flags.gpuMode
	}
	if flags.mainGPU >= 0 {
		options["main_gpu"] = strconv.Itoa(flags.mainGPU)
	}
	if flags.tensorSplit != "" {
		options["tensor_split"] = flags.tensorSplit
	}
	if flags.splitMode != "" {
		options["split_mode"] = flags.splitMode
	}
	if flags.flashAttention {
		options["flash_attn"] = "true"
	}
	if flags.adapterPath != "" {
		options["adapter_path"] = flags.adapterPath
	}
	if len(options) == 0 {
		return nil
	}
	return options
}

func validateRepeatPenalty(value float64) error {
	if value == 0 {
		return nil
	}
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 1 {
		return fmt.Errorf("repeat-penalty must be finite and >= 1")
	}
	return nil
}

func validateRuntimeOptions(options map[string]string) error {
	if len(options) == 0 {
		return nil
	}
	for _, key := range []string{"threads", "ctx_size", "gpu_layers", "main_gpu"} {
		if value := options[key]; value != "" {
			n, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("%s must be an integer", key)
			}
			if (key == "threads" || key == "ctx_size") && n <= 0 {
				return fmt.Errorf("%s must be greater than 0", key)
			}
			if (key == "gpu_layers" || key == "main_gpu") && n < 0 {
				return fmt.Errorf("%s must be greater than or equal to 0", key)
			}
		}
	}
	switch strings.ToLower(strings.TrimSpace(options["gpu"])) {
	case "", "off", "none", "false", "0", "auto", "full", "all", "true", "1":
	default:
		return fmt.Errorf("gpu must be one of off, auto, or full")
	}
	switch strings.ToLower(strings.TrimSpace(options["split_mode"])) {
	case "", "none", "layer", "row":
	default:
		return fmt.Errorf("split_mode must be one of none, layer, or row")
	}
	switch strings.ToLower(strings.TrimSpace(options["flash_attn"])) {
	case "", "1", "t", "true", "yes", "y", "on", "0", "f", "false", "no", "n", "off":
	default:
		return fmt.Errorf("flash_attn must be a boolean")
	}
	return nil
}

func runServe(args []string, stdout, stderr io.Writer) int {
	var backend string
	var addr string
	var modelID string
	var threads int
	var ctxSize int
	var gpuLayers int
	var gpuMode string
	var mainGPU int
	var tensorSplit string
	var splitMode string
	var flashAttention bool
	var native bool

	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "llama.cpp", "runtime backend")
	fs.BoolVar(&native, "native", false, "use the experimental pure-Go native backend")
	fs.StringVar(&addr, "addr", ":8080", "listen address")
	fs.StringVar(&modelID, "model", "nego-model", "served model id")
	fs.IntVar(&threads, "threads", 0, "llama.cpp CPU threads")
	fs.IntVar(&ctxSize, "ctx-size", 0, "llama.cpp context size")
	fs.IntVar(&gpuLayers, "gpu-layers", 0, "llama.cpp GPU layers")
	fs.StringVar(&gpuMode, "gpu", "", "llama.cpp GPU mode: off, auto, or full")
	fs.IntVar(&mainGPU, "main-gpu", -1, "llama.cpp main GPU index")
	fs.StringVar(&tensorSplit, "tensor-split", "", "llama.cpp comma-separated tensor split")
	fs.StringVar(&splitMode, "split-mode", "", "llama.cpp multi-GPU split mode")
	fs.BoolVar(&flashAttention, "flash-attn", false, "enable llama.cpp flash attention")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if native && flagWasSet(fs, "backend") {
		fmt.Fprintln(stderr, "nego: use either --native or --backend, not both")
		return 2
	}
	if native {
		backend = "native"
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego serve <model-path> [flags]")
		return 2
	}
	options := runtimeOptions(nil, runtimeFlagOptions{
		threads:        threads,
		ctxSize:        ctxSize,
		gpuLayers:      gpuLayers,
		gpuMode:        gpuMode,
		mainGPU:        mainGPU,
		tensorSplit:    tensorSplit,
		splitMode:      splitMode,
		flashAttention: flashAttention,
	})
	if err := validateRuntimeOptions(options); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Backend: backend,
		Path:    positionals[0],
		Options: options,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	defer model.Close()
	handler, err := server.NewHandler(server.HandlerOptions{ModelID: modelID, Model: model})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Serving %s on %s\n", modelID, addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	return 0
}

func runEmbed(args []string, stdout, stderr io.Writer) int {
	var backend string
	var endpoint string
	var modelID string
	var apiKey string

	fs := flag.NewFlagSet("embed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "openai-compatible", "runtime backend")
	fs.StringVar(&endpoint, "endpoint", "", "backend endpoint")
	fs.StringVar(&modelID, "model", "", "embedding model id")
	fs.StringVar(&apiKey, "api-key", "", "backend API key")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) < 1 {
		fmt.Fprintln(stderr, "usage: nego embed <text> --endpoint <url> --model <name> [flags]")
		return 2
	}
	if endpoint == "" || modelID == "" {
		fmt.Fprintln(stderr, "nego: --endpoint and --model are required")
		return 2
	}
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Backend:  backend,
		Endpoint: endpoint,
		Model:    modelID,
		APIKey:   apiKey,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	defer model.Close()
	resp, err := nego.Embed(context.Background(), model, nego.EmbeddingRequest{Input: []string{strings.Join(positionals, " ")}})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	_ = json.NewEncoder(stdout).Encode(resp)
	return 0
}

func runRuns(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		runsUsage(stderr)
		return 2
	}
	switch args[0] {
	case "list":
		return runRunsList(args[1:], stdout, stderr)
	case "show":
		return runRunsShow(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		runsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown runs command %q\n", args[0])
		runsUsage(stderr)
		return 2
	}
}

func runBackends(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		backendsUsage(stderr)
		return 2
	}
	switch args[0] {
	case "list":
		return runBackendsList(args[1:], stdout, stderr)
	case "info":
		return runBackendsInfo(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		backendsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown backends command %q\n", args[0])
		backendsUsage(stderr)
		return 2
	}
}

func runBackendsList(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("backends list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 0 {
		fmt.Fprintln(stderr, "usage: nego backends list [flags]")
		return 2
	}
	infos := nego.ListBackends()
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(infos)
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tCAPABILITIES\tREQUIRED")
	for _, info := range infos {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", info.Name, strings.Join(info.Capabilities, ", "), strings.Join(info.Required, ", "))
	}
	_ = tw.Flush()
	return 0
}

func runBackendsInfo(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("backends info", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego backends info <name> [flags]")
		return 2
	}
	info, ok := nego.BackendInfoByName(positionals[0])
	if !ok {
		fmt.Fprintf(stderr, "nego: backend %q is not registered\n", positionals[0])
		return 4
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(info)
		return 0
	}
	writeBackendInfo(stdout, info)
	return 0
}

func writeBackendInfo(w io.Writer, info nego.BackendInfo) {
	fmt.Fprintf(w, "Name:        %s\n", info.Name)
	if info.Description != "" {
		fmt.Fprintf(w, "Description: %s\n", info.Description)
	}
	if len(info.Capabilities) > 0 {
		fmt.Fprintf(w, "Capabilities:%s\n", " "+strings.Join(info.Capabilities, ", "))
	}
	if len(info.Required) > 0 {
		fmt.Fprintf(w, "Required:    %s\n", strings.Join(info.Required, ", "))
	}
	if len(info.Options) > 0 {
		fmt.Fprintln(w, "Options:")
		for _, option := range info.Options {
			if option.Description == "" {
				fmt.Fprintf(w, "  - %s\n", option.Name)
				continue
			}
			fmt.Fprintf(w, "  - %s: %s\n", option.Name, option.Description)
		}
	}
}

func backendsUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego backends list [flags]")
	fmt.Fprintln(w, "  nego backends info <name> [flags]")
}

func runDataset(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		datasetUsage(stderr)
		return 2
	}
	switch args[0] {
	case "inspect":
		return runDatasetInspect(args[1:], stdout, stderr)
	case "validate":
		return runDatasetValidate(args[1:], stdout, stderr)
	case "convert":
		return runDatasetConvert(args[1:], stdout, stderr)
	case "filter":
		return runDatasetFilter(args[1:], stdout, stderr)
	case "split":
		return runDatasetSplit(args[1:], stdout, stderr)
	case "sample":
		return runDatasetSample(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		datasetUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown dataset command %q\n", args[0])
		datasetUsage(stderr)
		return 2
	}
}

func runDatasetInspect(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("dataset inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego dataset inspect <file> [flags]")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	info := datasetSummary(positionals[0], rows)
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(info)
		return 0
	}
	fmt.Fprintf(stdout, "Path:        %s\n", info["path"])
	fmt.Fprintf(stdout, "Rows:        %d\n", info["rows"])
	fmt.Fprintf(stdout, "Columns:     %s\n", strings.Join(info["columns"].([]string), ", "))
	fmt.Fprintln(stdout, "Formats:")
	for key, value := range info["formats"].(map[string]int) {
		if value > 0 {
			fmt.Fprintf(stdout, "  %s: %d\n", key, value)
		}
	}
	return 0
}

func runDatasetValidate(args []string, stdout, stderr io.Writer) int {
	var format string
	fs := flag.NewFlagSet("dataset validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&format, "format", "auto", "dataset format: auto, chat, completion, or instruction")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego dataset validate <file> [flags]")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if err := datasets.ValidateFormat(rows, format); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Valid %s dataset: %d rows\n", format, len(rows))
	return 0
}

func runDatasetConvert(args []string, stdout, stderr io.Writer) int {
	var output string
	var format string
	var selectFields string
	var requireFields string
	fs := flag.NewFlagSet("dataset convert", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&output, "out", "", "output JSONL file, or - for stdout")
	fs.StringVar(&format, "format", "jsonl", "output format: jsonl")
	fs.StringVar(&selectFields, "select", "", "comma-separated fields to keep")
	fs.StringVar(&requireFields, "require", "", "comma-separated fields that must be present and non-empty")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || output == "" {
		fmt.Fprintln(stderr, "usage: nego dataset convert <file> --out <file> [flags]")
		return 2
	}
	if strings.ToLower(format) != "jsonl" {
		fmt.Fprintf(stderr, "nego: unsupported output format %q\n", format)
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	rows = datasets.RequireFields(rows, splitCommaFields(requireFields))
	rows = datasets.SelectFields(rows, splitCommaFields(selectFields))
	if err := writeDatasetRows(output, rows, stdout); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if output != "-" {
		fmt.Fprintf(stdout, "Converted: %d rows -> %s\n", len(rows), output)
	}
	return 0
}

func runDatasetFilter(args []string, stdout, stderr io.Writer) int {
	var output string
	var where repeatedFlag
	var selectFields string
	var requireFields string
	fs := flag.NewFlagSet("dataset filter", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&output, "out", "", "output JSONL file, or - for stdout")
	fs.Var(&where, "where", "filter expression, repeatable: field, field=value, field!=value, field~text, field!~text")
	fs.StringVar(&selectFields, "select", "", "comma-separated fields to keep")
	fs.StringVar(&requireFields, "require", "", "comma-separated fields that must be present and non-empty")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || output == "" {
		fmt.Fprintln(stderr, "usage: nego dataset filter <file> --where <expr> --out <file> [flags]")
		return 2
	}
	filters, err := parseDatasetFilters(where)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	rows = datasets.RequireFields(rows, splitCommaFields(requireFields))
	rows = datasets.FilterRows(rows, filters)
	rows = datasets.SelectFields(rows, splitCommaFields(selectFields))
	if err := writeDatasetRows(output, rows, stdout); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if output != "-" {
		fmt.Fprintf(stdout, "Filtered: %d rows -> %s\n", len(rows), output)
	}
	return 0
}

func runDatasetSplit(args []string, stdout, stderr io.Writer) int {
	var trainOut string
	var testOut string
	var testSize float64
	var seed int64
	fs := flag.NewFlagSet("dataset split", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&trainOut, "train-out", "", "output train JSONL file")
	fs.StringVar(&testOut, "test-out", "", "output test JSONL file")
	fs.Float64Var(&testSize, "test-size", 0.1, "test split ratio")
	fs.Int64Var(&seed, "seed", 42, "shuffle seed")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || trainOut == "" || testOut == "" {
		fmt.Fprintln(stderr, "usage: nego dataset split <file> --train-out <file> --test-out <file> [flags]")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if err := ensureDatasetOutputAvailable(trainOut, testOut); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	train, test := datasets.Split(rows, testSize, seed)
	if err := writeDatasetFile(trainOut, train); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if err := writeDatasetFile(testOut, test); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Train: %d rows -> %s\n", len(train), trainOut)
	fmt.Fprintf(stdout, "Test:  %d rows -> %s\n", len(test), testOut)
	return 0
}

func runDatasetSample(args []string, stdout, stderr io.Writer) int {
	var n int
	var seed int64
	fs := flag.NewFlagSet("dataset sample", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.IntVar(&n, "n", 10, "number of rows")
	fs.Int64Var(&seed, "seed", 42, "shuffle seed")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego dataset sample <file> [flags]")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if n < 0 {
		n = 0
	}
	copied := append([]datasets.Row(nil), rows...)
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(copied), func(i, j int) {
		copied[i], copied[j] = copied[j], copied[i]
	})
	if n > len(copied) {
		n = len(copied)
	}
	if err := datasets.WriteJSONL(stdout, copied[:n]); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	return 0
}

func datasetUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego dataset inspect <file> [flags]")
	fmt.Fprintln(w, "  nego dataset validate <file> [flags]")
	fmt.Fprintln(w, "  nego dataset convert <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset filter <file> --where <expr> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset split <file> --train-out <file> --test-out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset sample <file> [flags]")
}

func datasetSummary(path string, rows []datasets.Row) map[string]any {
	columns := make(map[string]bool)
	formats := map[string]int{"chat": 0, "completion": 0, "instruction": 0, "unknown": 0}
	for _, row := range rows {
		for key := range row {
			columns[key] = true
		}
		switch {
		case row["messages"] != nil:
			formats["chat"]++
		case row["prompt"] != nil && (row["completion"] != nil || row["response"] != nil):
			formats["completion"]++
		case row["instruction"] != nil && row["output"] != nil:
			formats["instruction"]++
		default:
			formats["unknown"]++
		}
	}
	columnList := make([]string, 0, len(columns))
	for column := range columns {
		columnList = append(columnList, column)
	}
	sort.Strings(columnList)
	return map[string]any{
		"path":    path,
		"rows":    len(rows),
		"columns": columnList,
		"formats": formats,
	}
}

func ensureDatasetOutputAvailable(paths ...string) error {
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("refusing to overwrite existing file %q", path)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func writeDatasetFile(path string, rows []datasets.Row) error {
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("refusing to overwrite existing file %q", path)
		}
		return err
	}
	defer file.Close()
	return datasets.WriteJSONL(file, rows)
}

func writeDatasetRows(path string, rows []datasets.Row, stdout io.Writer) error {
	if path == "-" {
		return datasets.WriteJSONL(stdout, rows)
	}
	return writeDatasetFile(path, rows)
}

func splitCommaFields(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	fields := make([]string, 0, len(parts))
	for _, part := range parts {
		field := strings.TrimSpace(part)
		if field != "" {
			fields = append(fields, field)
		}
	}
	return fields
}

func parseDatasetFilters(values []string) ([]datasets.Filter, error) {
	filters := make([]datasets.Filter, 0, len(values))
	for _, value := range values {
		filter, err := datasets.ParseFilter(value)
		if err != nil {
			return nil, err
		}
		filters = append(filters, filter)
	}
	return filters, nil
}

func runRunsList(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("runs list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego runs list <runs.jsonl> [flags]")
		return 2
	}
	entries, err := runs.Read(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(entries)
		return 0
	}
	if len(entries) == 0 {
		fmt.Fprintln(stdout, "No runs found.")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tTIME\tCOMMAND\tBACKEND\tMODEL/PATH\tSTATUS\tDURATION")
	for _, entry := range entries {
		status := "ok"
		if entry.Error != "" {
			status = "error"
		}
		target := entry.Model
		if target == "" {
			target = entry.Path
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%dms\n", entry.ID, entry.StartedAt.Format(time.RFC3339), entry.Command, entry.Backend, target, status, entry.DurationMS)
	}
	_ = tw.Flush()
	return 0
}

func runRunsShow(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("runs show", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 2 {
		fmt.Fprintln(stderr, "usage: nego runs show <runs.jsonl> <id> [flags]")
		return 2
	}
	entries, err := runs.Read(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	entry, ok := runs.Find(entries, positionals[1])
	if !ok {
		fmt.Fprintf(stderr, "nego: run %q not found\n", positionals[1])
		return 4
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(entry)
		return 0
	}
	writeRunInfo(stdout, entry)
	return 0
}

func runsUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego runs list <runs.jsonl> [flags]")
	fmt.Fprintln(w, "  nego runs show <runs.jsonl> <id> [flags]")
}

func writeRunInfo(w io.Writer, entry runs.Entry) {
	fmt.Fprintf(w, "ID:        %s\n", entry.ID)
	fmt.Fprintf(w, "Command:   %s\n", entry.Command)
	fmt.Fprintf(w, "Started:   %s\n", entry.StartedAt.Format(time.RFC3339))
	fmt.Fprintf(w, "Duration:  %dms\n", entry.DurationMS)
	if entry.Backend != "" {
		fmt.Fprintf(w, "Backend:   %s\n", entry.Backend)
	}
	if entry.Model != "" {
		fmt.Fprintf(w, "Model:     %s\n", entry.Model)
	}
	if entry.Path != "" {
		fmt.Fprintf(w, "Path:      %s\n", entry.Path)
	}
	if entry.Error != "" {
		fmt.Fprintf(w, "Error:     %s\n", entry.Error)
	}
	if entry.Prompt != "" {
		fmt.Fprintf(w, "Prompt:\n%s\n", entry.Prompt)
	}
	if len(entry.Messages) > 0 {
		fmt.Fprintln(w, "Messages:")
		for _, message := range entry.Messages {
			fmt.Fprintf(w, "  %s: %s\n", message.Role, message.Content)
		}
	}
	if entry.Output != "" {
		fmt.Fprintf(w, "Output:\n%s\n", entry.Output)
	}
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
	case "remove":
		return runModelsRemove(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		modelsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown models command %q\n", args[0])
		modelsUsage(stderr)
		return 2
	}
}

func runCache(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		cacheUsage(stderr)
		return 2
	}
	switch args[0] {
	case "usage":
		return runCacheUsage(args[1:], stdout, stderr)
	case "gc":
		return runCacheGC(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		cacheUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown cache command %q\n", args[0])
		cacheUsage(stderr)
		return 2
	}
}

func runCacheUsage(args []string, stdout, stderr io.Writer) int {
	var cacheDir string
	var jsonOutput bool
	fs := flag.NewFlagSet("cache usage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cacheDir, "cache-dir", "", "cache directory")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	resolvedCache, err := cache.ResolveCacheDir(cacheDir)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	size, files, err := dirUsage(resolvedCache)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	entries, err := registry.NewStore(resolvedCache).List()
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	usage := map[string]any{
		"cache_dir":         resolvedCache,
		"size":              size,
		"files":             files,
		"registered_models": len(entries),
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(usage)
		return 0
	}
	fmt.Fprintf(stdout, "Cache:             %s\n", resolvedCache)
	fmt.Fprintf(stdout, "Size:              %s\n", humanBytes(size))
	fmt.Fprintf(stdout, "Files:             %d\n", files)
	fmt.Fprintf(stdout, "Registered models: %d\n", len(entries))
	return 0
}

func runCacheGC(args []string, stdout, stderr io.Writer) int {
	var cacheDir string
	var yes bool
	fs := flag.NewFlagSet("cache gc", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cacheDir, "cache-dir", "", "cache directory")
	fs.BoolVar(&yes, "yes", false, "confirm cleanup")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !yes {
		fmt.Fprintln(stderr, "nego: refusing to clean cache without --yes")
		return 2
	}
	resolvedCache, err := cache.ResolveCacheDir(cacheDir)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	removed, err := removeTempCacheFiles(resolvedCache)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Removed %d temporary cache files\n", removed)
	return 0
}

func cacheUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego cache usage [flags]")
	fmt.Fprintln(w, "  nego cache gc --yes [flags]")
}

func dirUsage(root string) (int64, int, error) {
	var size int64
	var files int
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		size += info.Size()
		files++
		return nil
	})
	if errors.Is(err, os.ErrNotExist) {
		return 0, 0, nil
	}
	return size, files, err
}

func removeTempCacheFiles(root string) (int, error) {
	removed := 0
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		name := d.Name()
		if strings.HasPrefix(name, ".nego-") || strings.HasPrefix(name, ".registry-") || strings.HasSuffix(name, ".lock") {
			if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
			removed++
		}
		return nil
	})
	if errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}
	return removed, err
}

func runModelsRemove(args []string, stdout, stderr io.Writer) int {
	var cacheDir string
	var revision string
	var yes bool

	fs := flag.NewFlagSet("models remove", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cacheDir, "cache-dir", "", "cache directory")
	fs.StringVar(&revision, "revision", "", "revision to remove")
	fs.BoolVar(&yes, "yes", false, "confirm removal")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego models remove <repo-id> --yes [flags]")
		return 2
	}
	if !yes {
		fmt.Fprintln(stderr, "nego: refusing to remove model without --yes")
		return 2
	}

	resolvedCache, err := cache.ResolveCacheDir(cacheDir)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	store := registry.NewStore(resolvedCache)
	entry, err := store.Find(positionals[0], revision)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if err := removeManagedLocalFiles(entry); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if _, err := store.Remove(positionals[0], revision); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Removed %s", entry.RepoID)
	if entry.Revision != "" {
		fmt.Fprintf(stdout, "@%s", entry.Revision)
	}
	fmt.Fprintln(stdout)
	return 0
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
	fmt.Fprintln(w, "  nego models remove <repo-id> --yes [flags]")
}

func removeManagedLocalFiles(entry registry.Entry) error {
	if entry.LocalDir == "" || len(entry.Files) == 0 {
		return nil
	}
	localRoot, err := filepath.Abs(entry.LocalDir)
	if err != nil {
		return err
	}
	for _, file := range entry.Files {
		path, err := cache.SafeJoin(localRoot, file.Path)
		if err != nil {
			return err
		}
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		cleanupEmptyParents(filepath.Dir(path), localRoot)
	}
	return nil
}

func cleanupEmptyParents(start, root string) {
	current := start
	for {
		if current == root || current == "." || current == string(filepath.Separator) {
			return
		}
		if err := os.Remove(current); err != nil {
			return
		}
		current = filepath.Dir(current)
	}
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
