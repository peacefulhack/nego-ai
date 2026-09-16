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
	_ "github.com/gakon/nego-ai/backends/nativehf"
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
	case "status":
		return runStatus(args[1:], stdout, stderr)
	case "memory":
		return runMemory(args[1:], stdout, stderr)
	case "run":
		return runModel(args[1:], stdout, stderr)
	case "chat":
		return runChat(args[1:], stdin, stdout, stderr)
	case "config":
		return runConfig(args[1:], stdout, stderr)
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
	compatibility := ""
	if commonRepoType == hub.RepoTypeModel {
		compatibility = downloadCompatibilitySummary(path)
	}
	if jsonOutput {
		result := map[string]string{"path": path}
		if resolvedGGUF != nil {
			result["repo"] = resolvedGGUF.RepoID
			result["file"] = resolvedGGUF.Filename
		}
		if compatibility != "" {
			result["compatibility"] = compatibility
		}
		_ = json.NewEncoder(stdout).Encode(result)
	} else if quiet {
		fmt.Fprintln(stdout, path)
	} else {
		fmt.Fprintf(stdout, "Saved to: %s\n", path)
		if resolvedGGUF != nil {
			fmt.Fprintf(stdout, "Runtime file: %s/%s\n", resolvedGGUF.RepoID, resolvedGGUF.Filename)
		}
		if compatibility != "" {
			fmt.Fprintf(stdout, "Compatibility: %s\n", compatibility)
		}
	}
	return 0
}

func downloadCompatibilitySummary(path string) string {
	report, err := modelinfo.Check(path)
	if err != nil || report.Artifact == nil {
		return ""
	}
	run := firstNonEmptyString(report.Artifact.RecommendedRunBackend, "not ready")
	train := firstNonEmptyString(report.Artifact.RecommendedTrainBackend, "not ready")
	return fmt.Sprintf("format=%s run=%s train=%s", report.Artifact.Format, run, train)
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
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
	case "force", "local-files-only", "quiet", "json", "yes", "no-generation-prompt", "interactive", "flash-attn", "gguf", "native", "no-manifest", "dry-run":
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
	fmt.Fprintln(w, "  nego runs compare <runs.jsonl> <baseline-id> <candidate-id> [flags]")
	fmt.Fprintln(w, "  nego backends list [flags]")
	fmt.Fprintln(w, "  nego backends info <name> [flags]")
	fmt.Fprintln(w, "  nego dataset inspect <file> [flags]")
	fmt.Fprintln(w, "  nego dataset quality <file> [flags]")
	fmt.Fprintln(w, "  nego dataset validate <file> [flags]")
	fmt.Fprintln(w, "  nego dataset tokens <file> --model <model-path> [flags]")
	fmt.Fprintln(w, "  nego dataset render <file> [flags]")
	fmt.Fprintln(w, "  nego dataset convert <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset filter <file> --where <expr> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset dedupe <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset shuffle <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset split <file> --train-out <file> --test-out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset sample <file> [flags]")
	fmt.Fprintln(w, "  nego cache usage [flags]")
	fmt.Fprintln(w, "  nego cache gc --yes [flags]")
	fmt.Fprintln(w, "  nego tokenize <model-path> <text> [flags]")
	fmt.Fprintln(w, "  nego tokenize check <model-path> <fixtures.json> [flags]")
	fmt.Fprintln(w, "  nego tokens <model-path> <text>")
	fmt.Fprintln(w, "  nego context <model-path> <text> [flags]")
	fmt.Fprintln(w, "  nego prompt <model-path> --user <text> [flags]")
	fmt.Fprintln(w, "  nego inspect <model-path> [flags]")
	fmt.Fprintln(w, "  nego check <model-path> [flags]")
	fmt.Fprintln(w, "  nego status <model-path> [flags]")
	fmt.Fprintln(w, "  nego memory <model-path> [flags]")
	fmt.Fprintln(w, "  nego run <model-path> <prompt> [flags]")
	fmt.Fprintln(w, "  nego chat <model-path> <message> [flags]")
	fmt.Fprintln(w, "  nego config run <model-path> --out <config.json> [flags]")
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
	fmt.Fprintln(w, "  nego train capabilities <model-path> [flags]")
	fmt.Fprintln(w, "  nego train native <model-path> --train-file <file> [--out <dir>] [flags]")
	fmt.Fprintln(w, "  nego train <job.json> [flags]")
	fmt.Fprintln(w, "  nego share manifest <model-dir> --out <file> [flags]")
	fmt.Fprintln(w, "  nego share check <model-dir> [flags]")
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
	case "check":
		return runShareCheck(args[1:], stdout, stderr)
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

func runConfig(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		configUsage(stderr)
		return 2
	}
	switch args[0] {
	case "run":
		return runConfigRun(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		configUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown config command %q\n", args[0])
		configUsage(stderr)
		return 2
	}
}

func runConfigRun(args []string, stdout, stderr io.Writer) int {
	var out string
	var prompt string
	var system string
	var logPath string
	var maxTokens int
	var temperature float64
	var topK int
	var topP float64
	var repeatPenalty float64
	var seed int64
	var jsonOutput bool
	fs := flag.NewFlagSet("config run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&out, "out", "", "runtime config output path")
	fs.StringVar(&prompt, "prompt", "Hello", "default prompt for nego run/chat")
	fs.StringVar(&system, "system", "", "default system message for nego chat")
	fs.StringVar(&logPath, "log", "", "default JSONL run log path")
	fs.IntVar(&maxTokens, "max-tokens", 0, "maximum tokens to generate")
	fs.Float64Var(&temperature, "temperature", 0, "sampling temperature")
	fs.IntVar(&topK, "top-k", 0, "keep only the top k tokens during sampling")
	fs.Float64Var(&topP, "top-p", 0, "nucleus sampling probability")
	fs.Float64Var(&repeatPenalty, "repeat-penalty", 0, "penalty for repeated tokens, >= 1")
	fs.Int64Var(&seed, "seed", 0, "random seed")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON config to stdout")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || (out == "" && !jsonOutput) {
		fmt.Fprintln(stderr, "usage: nego config run <model-path> --out <config.json> [flags]")
		return 2
	}
	if err := validateTopK(topK); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if err := validateRepeatPenalty(repeatPenalty); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	cfg, err := buildRuntimeConfig(positionals[0], runtimeConfigOverrides{
		prompt:        prompt,
		system:        system,
		logPath:       logPath,
		maxTokens:     maxTokens,
		temperature:   temperature,
		topK:          topK,
		topP:          topP,
		repeatPenalty: repeatPenalty,
		seed:          seed,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if out != "" {
		if err := writeRuntimeConfig(out, cfg); err != nil {
			fmt.Fprintf(stderr, "nego: write config: %v\n", err)
			return 1
		}
	}
	if jsonOutput {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		_ = encoder.Encode(cfg)
		return 0
	}
	fmt.Fprintf(stdout, "Runtime config: %s\n", out)
	fmt.Fprintf(stdout, "Backend:        %s\n", cfg.Backend)
	fmt.Fprintf(stdout, "Model path:     %s\n", cfg.Path)
	fmt.Fprintf(stdout, "Run:            nego run -f %s\n", out)
	fmt.Fprintf(stdout, "Chat:           nego chat -f %s\n", out)
	return 0
}

func configUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego config run <model-path> --out <config.json> [flags]")
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
	if manifest.ArtifactType != "" {
		fmt.Fprintf(stdout, "Artifact:       %s\n", manifest.ArtifactType)
	}
	if manifest.BaseModel != "" {
		fmt.Fprintf(stdout, "Base model:     %s\n", manifest.BaseModel)
	}
	fmt.Fprintf(stdout, "Files:          %d\n", len(manifest.Files))
	fmt.Fprintf(stdout, "Size:           %s\n", humanBytes(manifest.TotalSize))
	return 0
}

func runShareCheck(args []string, stdout, stderr io.Writer) int {
	var repo string
	var baseModel string
	var maxInlineSize int64
	var jsonOutput bool
	fs := flag.NewFlagSet("share check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&repo, "repo", "", "target repository id")
	fs.StringVar(&baseModel, "base-model", "", "base model id")
	fs.Int64Var(&maxInlineSize, "max-inline-size", 0, "maximum inline file size in bytes")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON report")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego share check <model-dir> [flags]")
		return 2
	}
	report, err := share.Check(share.CheckOptions{
		Path:          positionals[0],
		RepoID:        repo,
		BaseModel:     baseModel,
		MaxInlineSize: maxInlineSize,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(report)
	} else {
		printShareCheck(stdout, report)
	}
	if !report.Valid {
		return 1
	}
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

func printShareCheck(w io.Writer, report *share.CheckReport) {
	fmt.Fprintln(w, "Share check")
	fmt.Fprintf(w, "Path:          %s\n", report.Path)
	if report.RepoID != "" {
		fmt.Fprintf(w, "Repo:          %s\n", report.RepoID)
	}
	if report.BaseModel != "" {
		fmt.Fprintf(w, "Base model:    %s\n", report.BaseModel)
	}
	if report.ArtifactType != "" {
		fmt.Fprintf(w, "Artifact:      %s\n", report.ArtifactType)
	}
	if report.NativeAdapter != nil {
		fmt.Fprintln(w, "Native adapter:")
		if report.NativeAdapter.RecommendedBackend != "" {
			fmt.Fprintf(w, "  Backend:        %s\n", report.NativeAdapter.RecommendedBackend)
		}
		if report.NativeAdapter.Method != "" {
			fmt.Fprintf(w, "  Method:         %s\n", report.NativeAdapter.Method)
		}
		if report.NativeAdapter.UpdatedTokens > 0 {
			fmt.Fprintf(w, "  Updated tokens: %d\n", report.NativeAdapter.UpdatedTokens)
		}
		if report.NativeAdapter.EvalCoverage != nil {
			printNativeAdapterEvalCoverage(w, report.NativeAdapter.EvalCoverage)
		}
		if report.NativeAdapter.Tokenizer != nil {
			writeTokenizerReport(w, report.NativeAdapter.Tokenizer)
		}
	}
	fmt.Fprintf(w, "Files:         %d\n", report.Files)
	fmt.Fprintf(w, "Size:          %s\n", humanBytes(report.TotalSize))
	fmt.Fprintf(w, "Inline limit:  %s\n", humanBytes(report.MaxInlineSize))
	fmt.Fprintf(w, "Inline files:  %d\n", report.InlineFiles)
	fmt.Fprintf(w, "Large files:   %d\n", len(report.LargeFiles))
	for _, file := range report.LargeFiles {
		fmt.Fprintf(w, "  - %s %s requires LFS/Xet\n", file.Path, humanBytes(file.Size))
	}
	for _, warning := range report.Warnings {
		fmt.Fprintf(w, "Warning:       %s\n", warning)
	}
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
	fmt.Fprintln(w, "  nego share check <model-dir> [flags]")
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
		case "capabilities":
			return runTrainCapabilities(args[1:], stdout, stderr)
		case "native":
			return runTrainNative(args[1:], stdout, stderr)
		case "latest":
			return runTrainLatest(args[1:], stdout, stderr)
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
	var maxContext int
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
	fs.IntVar(&maxContext, "max-context", 0, "maximum context tokens checked during preflight")
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
		MaxContext:    maxContext,
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

func runTrainCapabilities(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("train capabilities", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON report")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego train capabilities <model-path> [flags]")
		return 2
	}
	report, err := training.Assess(positionals[0])
	if jsonOutput {
		body := map[string]any{
			"success": err == nil,
			"report":  report,
		}
		if err != nil {
			body["error"] = err.Error()
		}
		_ = json.NewEncoder(stdout).Encode(body)
	} else if err == nil {
		printTrainingAssessment(stdout, report)
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
	var maxContext int
	var jsonOutput bool
	var dryRun bool
	var logPath string
	var tokenizeCheckPath string
	var dedupeKeys repeatedFlag
	var dedupeTrim bool
	var dedupeFold bool
	var failOnDupes bool
	fs := flag.NewFlagSet("train native", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&trainFile, "train-file", "", "training dataset file")
	fs.StringVar(&evalFile, "eval-file", "", "evaluation dataset file")
	fs.StringVar(&datasetFormat, "dataset-format", "auto", "dataset format: auto, chat, completion, or instruction")
	fs.StringVar(&outputDir, "out", "", "output adapter directory")
	fs.StringVar(&method, "method", "token-bias", "native training method")
	fs.Float64Var(&learningRate, "learning-rate", 0.1, "adapter learning-rate scale")
	fs.IntVar(&epochs, "epochs", 1, "number of passes over the dataset")
	fs.IntVar(&maxContext, "max-context", 0, "maximum context tokens checked before native training")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	fs.BoolVar(&dryRun, "dry-run", false, "validate and preview native training without writing output files")
	fs.StringVar(&logPath, "log", "", "append training result to JSONL run log")
	fs.StringVar(&tokenizeCheckPath, "tokenize-check", "", "tokenizer fixture JSON file to validate before native training")
	fs.Var(&dedupeKeys, "dedupe-key", "field used to warn about duplicate training rows, repeatable")
	fs.BoolVar(&dedupeTrim, "dedupe-trim-space", false, "trim string fields before duplicate training row checks")
	fs.BoolVar(&dedupeFold, "dedupe-ignore-case", false, "case-fold string fields before duplicate training row checks")
	fs.BoolVar(&failOnDupes, "fail-on-duplicates", false, "fail native training when duplicate rows are found")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || trainFile == "" || (outputDir == "" && !dryRun) {
		fmt.Fprintln(stderr, "usage: nego train native <model-path> --train-file <file> [--out <dir>] [flags]")
		return 2
	}
	var progress func(training.NativeProgress)
	if !jsonOutput {
		progress = nativeTrainingProgressPrinter(stderr)
	}
	var tokenizeCheck tokenizeCheckResult
	if tokenizeCheckPath != "" {
		var err error
		tokenizeCheck, err = runTrainingTokenizeCheck(positionals[0], tokenizeCheckPath, jsonOutput, stderr)
		if err != nil {
			if jsonOutput {
				_ = json.NewEncoder(stdout).Encode(map[string]any{
					"success":        false,
					"error":          err.Error(),
					"tokenize_check": tokenizeCheck,
				})
			} else {
				fmt.Fprintf(stderr, "nego: %v\n", err)
			}
			return 1
		}
	}
	started := time.Now().UTC()
	result, err := training.RunNative(context.Background(), training.NativeOptions{
		BaseModel:     positionals[0],
		TrainFile:     trainFile,
		EvalFile:      evalFile,
		DatasetFormat: datasetFormat,
		OutputDir:     outputDir,
		Method:        method,
		LearningRate:  learningRate,
		Epochs:        epochs,
		MaxContext:    maxContext,
		DryRun:        dryRun,
		Progress:      progress,
		DedupeKeys:    expandRepeatedCommaFields(dedupeKeys),
		DedupeTrim:    dedupeTrim,
		DedupeFold:    dedupeFold,
		FailOnDupes:   failOnDupes,
	})
	if logPath != "" {
		entry := nativeTrainingLogEntry(result, positionals[0], trainFile, evalFile, datasetFormat, outputDir, method, learningRate, epochs, maxContext, dryRun, started, err)
		if logErr := appendRuntimeLog(logPath, entry); logErr != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
			if err == nil {
				return 1
			}
		}
	}
	if jsonOutput {
		body := map[string]any{
			"success": err == nil,
			"result":  result,
		}
		if tokenizeCheckPath != "" {
			body["tokenize_check"] = tokenizeCheck
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

func runTrainingTokenizeCheck(modelPath, fixturePath string, jsonOutput bool, stderr io.Writer) (tokenizeCheckResult, error) {
	if !jsonOutput {
		fmt.Fprintf(stderr, "Checking tokenizer fixtures: %s\n", fixturePath)
	}
	tok, err := loadCLITokenizer(modelPath)
	if err != nil {
		return tokenizeCheckResult{}, err
	}
	cases, err := readTokenizeCheckCases(fixturePath)
	if err != nil {
		return tokenizeCheckResult{}, err
	}
	result := runTokenizeCheckCases(tok, cases)
	if result.Failed > 0 {
		if !jsonOutput {
			writeTokenizeCheckResult(stderr, result)
		}
		return result, fmt.Errorf("tokenizer check failed: %d/%d cases failed", result.Failed, result.Total)
	}
	if !jsonOutput {
		fmt.Fprintf(stderr, "Tokenizer check passed: %d/%d\n", result.Passed, result.Total)
	}
	return result, nil
}

func nativeTrainingLogEntry(result training.NativeResult, baseModel, trainFile, evalFile, datasetFormat, outputDir, method string, learningRate float64, epochs, maxContext int, dryRun bool, started time.Time, trainErr error) runs.Entry {
	artifact := ""
	if result.Artifact != nil {
		artifact = string(result.Artifact.Format)
	}
	topTokenIDs := make([]int, 0, len(result.TopTokens))
	for _, token := range result.TopTokens {
		topTokenIDs = append(topTokenIDs, token.ID)
	}
	entry := runs.Entry{
		ID:         runs.NewID(),
		Command:    "train native",
		Backend:    "native",
		Path:       baseModel,
		StartedAt:  started,
		DurationMS: time.Since(started).Milliseconds(),
		Training: &runs.Training{
			Method:        firstNonEmptyString(result.Method, method),
			TrainFile:     trainFile,
			EvalFile:      evalFile,
			DatasetFormat: firstNonEmptyString(result.DatasetFormat, datasetFormat),
			OutputDir:     firstNonEmptyString(result.OutputDir, outputDir),
			AdapterPath:   result.AdapterPath,
			ManifestPath:  result.ManifestPath,
			Artifact:      artifact,
			DryRun:        dryRun || result.DryRun,
			LearningRate:  result.LearningRate,
			Epochs:        firstPositive(result.Epochs, epochs),
			MaxContext:    firstPositive(result.MaxContext, maxContext),
			TrainRows:     result.TrainRows,
			EvalRows:      result.EvalRows,
			DuplicateRows: result.DuplicateRows,
			VocabSize:     result.VocabSize,
			EvalCoverage:  trainingRunEvalCoverage(result.EvalCoverage),
			TopTokenIDs:   topTokenIDs,
		},
	}
	if trainErr != nil {
		entry.Error = trainErr.Error()
	}
	return entry
}

func trainingRunEvalCoverage(summary *training.NativeEvalSummary) *runs.EvalCoverage {
	if summary == nil {
		return nil
	}
	return &runs.EvalCoverage{
		Rows:                summary.Rows,
		Tokens:              summary.Tokens,
		CoveredTokens:       summary.CoveredTokens,
		Coverage:            summary.Coverage,
		UniqueTokens:        summary.UniqueTokens,
		CoveredUniqueTokens: summary.CoveredUniqueTokens,
		UniqueCoverage:      summary.UniqueCoverage,
	}
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func nativeTrainingProgressPrinter(w io.Writer) func(training.NativeProgress) {
	seen := make(map[string]bool)
	return func(event training.NativeProgress) {
		if event.Stage == "" {
			return
		}
		key := event.Stage
		if event.Stage == "train" {
			key = fmt.Sprintf("%s:%d:%d", event.Stage, event.Epoch, event.RowsDone)
			if event.RowsDone > 0 && event.RowsDone < event.RowsTotal && event.RowsDone%100 != 0 {
				return
			}
		}
		if seen[key] {
			return
		}
		seen[key] = true
		message := event.Message
		if message == "" {
			message = event.Stage
		}
		if event.Stage == "train" && event.RowsTotal > 0 {
			if event.RowsDone > 0 {
				fmt.Fprintf(w, "Training native adapter: epoch %d/%d rows %d/%d\n", event.Epoch, event.Epochs, event.RowsDone, event.RowsTotal)
				return
			}
			fmt.Fprintf(w, "Training native adapter: epoch %d/%d rows 0/%d\n", event.Epoch, event.Epochs, event.RowsTotal)
			return
		}
		fmt.Fprintf(w, "%s...\n", sentenceCase(message))
	}
}

func sentenceCase(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return value
	}
	return strings.ToUpper(value[:1]) + value[1:]
}

func printNativeTrainingResult(w io.Writer, result training.NativeResult) {
	if result.DryRun {
		fmt.Fprintln(w, "Native training dry run passed")
	} else {
		fmt.Fprintln(w, "Native training completed")
	}
	fmt.Fprintf(w, "Base model:     %s\n", result.BaseModel)
	fmt.Fprintf(w, "Train file:     %s (%d rows)\n", result.TrainFile, result.TrainRows)
	if result.EvalFile != "" {
		fmt.Fprintf(w, "Eval file:      %s (%d rows)\n", result.EvalFile, result.EvalRows)
	}
	fmt.Fprintf(w, "Dataset format: %s\n", result.DatasetFormat)
	fmt.Fprintf(w, "Method:         %s\n", result.Method)
	fmt.Fprintf(w, "Epochs:         %d\n", result.Epochs)
	if result.MaxContext > 0 {
		fmt.Fprintf(w, "Max context:    %d\n", result.MaxContext)
	}
	if result.DuplicateRows > 0 {
		fmt.Fprintf(w, "Duplicates:     %d\n", result.DuplicateRows)
	}
	if result.TrainBudget != nil {
		printTrainingTokenBudget(w, "Train budget", result.TrainBudget)
	}
	if result.EvalBudget != nil {
		printTrainingTokenBudget(w, "Eval budget", result.EvalBudget)
	}
	if result.EvalCoverage != nil {
		printNativeEvalCoverage(w, result.EvalCoverage)
	}
	fmt.Fprintf(w, "Train tokens:   %d\n", result.TrainTokens)
	fmt.Fprintf(w, "Updated tokens: %d\n", result.UpdatedTokens)
	if result.Memory != nil {
		writeCheckMemory(w, result.Memory)
	}
	printNativeTrainingTopTokens(w, result.TopTokens)
	if result.DryRun {
		if result.AdapterPath != "" {
			fmt.Fprintf(w, "Planned adapter: %s\n", result.AdapterPath)
		}
		fmt.Fprintln(w, "Writes:         no")
	} else {
		fmt.Fprintf(w, "Adapter:        %s\n", result.AdapterPath)
	}
	if result.ManifestPath != "" {
		fmt.Fprintf(w, "Manifest:       %s\n", result.ManifestPath)
	}
	if result.ReadmePath != "" {
		fmt.Fprintf(w, "Guide:          %s\n", result.ReadmePath)
	}
	if runCommand := nativeTrainingRunCommand(result); runCommand != "" {
		fmt.Fprintf(w, "Run:            %s\n", runCommand)
	}
	if chatCommand := nativeTrainingChatCommand(result); chatCommand != "" {
		fmt.Fprintf(w, "Chat:           %s\n", chatCommand)
	}
	if shareCommand := nativeTrainingShareCommand(result); shareCommand != "" {
		fmt.Fprintf(w, "Share:          %s\n", shareCommand)
	}
	for _, warning := range result.Warnings {
		fmt.Fprintf(w, "Warning:        %s\n", warning)
	}
}

func printNativeEvalCoverage(w io.Writer, summary *training.NativeEvalSummary) {
	fmt.Fprintf(w, "Eval coverage:  %d/%d tokens (%.2f%%), %d/%d unique (%.2f%%)\n",
		summary.CoveredTokens,
		summary.Tokens,
		summary.Coverage*100,
		summary.CoveredUniqueTokens,
		summary.UniqueTokens,
		summary.UniqueCoverage*100,
	)
}

func nativeTrainingRunCommand(result training.NativeResult) string {
	args := nativeTrainingRuntimeArgs("run", result)
	if len(args) == 0 {
		return ""
	}
	args = append(args, "Hello")
	return formatCommand(args)
}

func nativeTrainingChatCommand(result training.NativeResult) string {
	args := nativeTrainingRuntimeArgs("chat", result)
	if len(args) == 0 {
		return ""
	}
	args = append(args, "Hello")
	return formatCommand(args)
}

func nativeTrainingShareCommand(result training.NativeResult) string {
	if result.DryRun || result.OutputDir == "" || result.ManifestPath == "" {
		return ""
	}
	return formatCommand([]string{"nego", "share", "package", result.OutputDir, "--out", result.OutputDir + ".zip"})
}

func nativeTrainingRuntimeArgs(command string, result training.NativeResult) []string {
	if result.BaseModel == "" || result.AdapterPath == "" {
		return nil
	}
	if result.OutputDir != "" && result.ManifestPath != "" {
		return []string{"nego", command, result.OutputDir}
	}
	args := []string{"nego", command}
	switch {
	case result.Artifact != nil && result.Artifact.Format == modelinfo.ArtifactFormatHFSafetensors:
		args = append(args, result.BaseModel, "--adapter", result.AdapterPath)
	default:
		args = append(args, "--native", result.BaseModel, "--adapter", result.AdapterPath)
	}
	return args
}

func formatCommand(args []string) string {
	if len(args) == 0 {
		return ""
	}
	quoted := make([]string, len(args))
	for i, arg := range args {
		quoted[i] = quoteCommandArg(arg)
	}
	return strings.Join(quoted, " ")
}

func quoteCommandArg(arg string) string {
	if arg == "" {
		return `""`
	}
	if strings.ContainsAny(arg, " \t\r\n\"") {
		return strconv.Quote(arg)
	}
	return arg
}

func printTrainingAssessment(w io.Writer, report training.Assessment) {
	fmt.Fprintln(w, "Training capabilities")
	fmt.Fprintf(w, "Base model:     %s\n", report.BaseModel)
	if report.Artifact != nil {
		fmt.Fprintf(w, "Format:         %s\n", report.Artifact.Format)
		if report.Artifact.ModelType != "" {
			fmt.Fprintf(w, "Model type:     %s\n", report.Artifact.ModelType)
		}
	}
	if report.Memory != nil {
		writeCheckMemory(w, report.Memory)
	}
	if report.Tokenizer != nil {
		writeTokenizerReport(w, report.Tokenizer)
	}
	fmt.Fprintln(w, "Methods:")
	for _, method := range report.Methods {
		status := string(method.Status)
		if method.Available && method.Status == "" {
			status = "ready"
		}
		fmt.Fprintf(w, "  - %-16s %s", method.Method, status)
		if method.Reason != "" {
			fmt.Fprintf(w, " (%s)", method.Reason)
		}
		fmt.Fprintln(w)
	}
	for _, warning := range report.Warnings {
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
	if report.MaxContext > 0 {
		fmt.Fprintf(w, "Max context:    %d\n", report.MaxContext)
	}
	if report.TrainTokens != nil {
		printTrainingTokenBudget(w, "Train tokens", report.TrainTokens)
	}
	if report.EvalTokens != nil {
		printTrainingTokenBudget(w, "Eval tokens", report.EvalTokens)
	}
	if report.Command != "" {
		command := strings.TrimSpace(strings.Join(append([]string{report.Command}, redactTrainingArgs(report.Args)...), " "))
		fmt.Fprintf(w, "Command:        %s\n", command)
	}
	for _, warning := range report.Warnings {
		fmt.Fprintf(w, "Warning:        %s\n", warning)
	}
}

func printTrainingTokenBudget(w io.Writer, label string, summary *training.TokenBudgetSummary) {
	fmt.Fprintf(w, "%s:   min %d / avg %.1f / max %d / total %d", label, summary.MinTokens, summary.AverageTokens, summary.MaxTokens, summary.TotalTokens)
	if summary.MaxContext > 0 {
		fmt.Fprintf(w, " / over %d", summary.OverLimit)
	}
	fmt.Fprintln(w)
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

func runTrainLatest(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	var baseModel string
	fs := flag.NewFlagSet("train latest", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	fs.StringVar(&baseModel, "base-model", "", "only include native training outputs for this base model")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) > 1 {
		fmt.Fprintln(stderr, "usage: nego train latest [outputs-dir] [flags]")
		return 2
	}
	root := "./outputs"
	if len(positionals) == 1 {
		root = positionals[0]
	}
	entry, err := training.LatestNativeManifest(training.NativeManifestDiscoveryOptions{
		Root:      root,
		BaseModel: baseModel,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 4
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(entry)
		return 0
	}
	writeLatestNativeTraining(stdout, entry)
	return 0
}

func writeLatestNativeTraining(w io.Writer, entry training.NativeManifestEntry) {
	manifest := entry.Manifest
	fmt.Fprintln(w, "Latest native training output")
	fmt.Fprintf(w, "Path:          %s\n", entry.Path)
	fmt.Fprintf(w, "Manifest:      %s\n", entry.ManifestPath)
	fmt.Fprintf(w, "Base model:    %s\n", manifest.BaseModel)
	if manifest.Method != "" {
		fmt.Fprintf(w, "Method:        %s\n", manifest.Method)
	}
	if manifest.RecommendedBackend != "" {
		fmt.Fprintf(w, "Backend:       %s\n", manifest.RecommendedBackend)
	}
	if manifest.AdapterPath != "" {
		adapterPath := manifest.AdapterPath
		if !filepath.IsAbs(adapterPath) {
			adapterPath = filepath.Join(entry.Path, filepath.FromSlash(adapterPath))
		}
		fmt.Fprintf(w, "Adapter:       %s\n", adapterPath)
	}
	if !entry.CreatedAt.IsZero() {
		fmt.Fprintf(w, "Created:       %s\n", entry.CreatedAt.Format(time.RFC3339))
	} else if !entry.ModifiedAt.IsZero() {
		fmt.Fprintf(w, "Modified:      %s\n", entry.ModifiedAt.Format(time.RFC3339))
	}
	if manifest.TrainRows > 0 || manifest.EvalRows > 0 {
		fmt.Fprintf(w, "Rows:          train %d / eval %d\n", manifest.TrainRows, manifest.EvalRows)
	}
	if manifest.EvalCoverage != nil {
		fmt.Fprintf(w, "Eval coverage: %d/%d tokens (%.2f%%)\n", manifest.EvalCoverage.CoveredTokens, manifest.EvalCoverage.Tokens, manifest.EvalCoverage.Coverage*100)
	}
	fmt.Fprintln(w, "Run:")
	fmt.Fprintf(w, "  nego run %s \"Hello\" --max-tokens 32\n", quoteCommandArg(entry.Path))
	fmt.Fprintln(w, "Chat:")
	fmt.Fprintf(w, "  nego chat %s \"Hello\" --max-tokens 32\n", quoteCommandArg(entry.Path))
}

func trainUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego train init --base-model <dir> --train-file <file> --out <job.json> [flags]")
	fmt.Fprintln(w, "  nego train check <job.json> [flags]")
	fmt.Fprintln(w, "  nego train validate <job.json> [flags]")
	fmt.Fprintln(w, "  nego train capabilities <model-path> [flags]")
	fmt.Fprintln(w, "  nego train native <model-path> --train-file <file> [--out <dir>] [flags]")
	fmt.Fprintln(w, "  nego train latest [outputs-dir] [flags]")
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
	if len(args) > 0 && args[0] == "check" {
		return runTokenizeCheck(args[1:], stdout, stderr)
	}
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
	tok, err := loadCLITokenizer(positionals[0])
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
	tok, err := loadCLITokenizer(args[0])
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
	tok, err := loadCLITokenizer(modelPath)
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

type textTokenizer interface {
	Encode(text string) ([]int, error)
	Decode(ids []int) (string, error)
	Count(text string) (int, error)
}

type ggufTextTokenizer struct {
	vocab *modelinfo.GGUFVocab
}

func (t ggufTextTokenizer) Encode(text string) ([]int, error) {
	return t.vocab.Encode(text, t.vocab.DefaultEncodeOptions())
}

func (t ggufTextTokenizer) Count(text string) (int, error) {
	ids, err := t.Encode(text)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

func (t ggufTextTokenizer) Decode(ids []int) (string, error) {
	return t.vocab.Decode(ids, modelinfo.DecodeOptions{SkipSpecial: true})
}

func loadCLITokenizer(path string) (textTokenizer, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		if strings.EqualFold(filepath.Ext(path), ".gguf") {
			return loadGGUFCLITokenizer(path)
		}
		return tokenizer.Load(path)
	}
	tokenizerPath := filepath.Join(path, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); err == nil {
		return tokenizer.Load(path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if runtimeFile, err := modelinfo.FindRuntimeFile(path, "gguf"); err == nil {
		return loadGGUFCLITokenizer(runtimeFile.Path)
	}
	return tokenizer.Load(path)
}

func loadGGUFCLITokenizer(path string) (textTokenizer, error) {
	vocab, err := modelinfo.InspectGGUFVocab(path)
	if err != nil {
		return nil, err
	}
	if len(vocab.Tokens) == 0 {
		return nil, fmt.Errorf("GGUF tokenizer %s does not contain tokens", path)
	}
	return ggufTextTokenizer{vocab: vocab}, nil
}

const maxTokenizeCheckBytes = 8 << 20

type tokenizeCheckCase = tokenizer.CheckCase
type tokenizeCheckCaseResult = tokenizer.CheckCaseResult
type tokenizeCheckResult = tokenizer.CheckResult

func runTokenizeCheck(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("tokenize check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 2 {
		fmt.Fprintln(stderr, "usage: nego tokenize check <model-path> <fixtures.json> [flags]")
		return 2
	}
	tok, err := loadCLITokenizer(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	cases, err := readTokenizeCheckCases(positionals[1])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	result := runTokenizeCheckCases(tok, cases)
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(result)
	} else {
		writeTokenizeCheckResult(stdout, result)
	}
	if result.Failed > 0 {
		return 1
	}
	return 0
}

func readTokenizeCheckCases(path string) ([]tokenizeCheckCase, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxTokenizeCheckBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxTokenizeCheckBytes {
		return nil, fmt.Errorf("tokenize check fixture %s exceeds %d bytes", path, maxTokenizeCheckBytes)
	}
	var cases []tokenizeCheckCase
	if err := json.Unmarshal(data, &cases); err == nil {
		return tokenizer.ValidateCheckCases(cases)
	}
	var wrapped struct {
		Cases []tokenizeCheckCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &wrapped); err != nil {
		return nil, fmt.Errorf("parse tokenize check fixture: %w", err)
	}
	return tokenizer.ValidateCheckCases(wrapped.Cases)
}

func runTokenizeCheckCases(tok textTokenizer, cases []tokenizeCheckCase) tokenizeCheckResult {
	return tokenizer.CheckCases(tok, cases)
}

func writeTokenizeCheckResult(w io.Writer, result tokenizeCheckResult) {
	fmt.Fprintln(w, "Tokenizer check")
	fmt.Fprintf(w, "Passed: %d\n", result.Passed)
	fmt.Fprintf(w, "Failed: %d\n", result.Failed)
	fmt.Fprintf(w, "Total:  %d\n", result.Total)
	if result.Failed == 0 {
		return
	}
	fmt.Fprintln(w, "Failures:")
	for _, c := range result.Cases {
		if c.Passed {
			continue
		}
		fmt.Fprintf(w, "  - %s\n", c.Name)
		for _, message := range c.Errors {
			fmt.Fprintf(w, "    - %s\n", message)
		}
	}
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

func runMemory(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	var contextLength uint64
	var kvBytes uint64
	fs := flag.NewFlagSet("memory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON estimate")
	fs.Uint64Var(&contextLength, "context", 0, "context length for KV cache estimate")
	fs.Uint64Var(&kvBytes, "kv-bytes", 4, "bytes per KV cache value")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego memory <model-path> [flags]")
		return 2
	}
	estimate, err := modelinfo.EstimateMemory(positionals[0], modelinfo.MemoryOptions{
		ContextLength: contextLength,
		KVBytes:       kvBytes,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(estimate)
		return 0
	}
	printMemoryEstimate(stdout, estimate)
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
	if info.Generation != nil {
		writeGenerationConfig(w, info.Generation)
	}
	if info.NativeAdapter != nil {
		writeNativeAdapterInfo(w, info.NativeAdapter)
	}
	if info.HFSpec != nil {
		fmt.Fprintln(w, "HF spec:")
		fmt.Fprintf(w, "  Ready:        %v\n", info.HFSpec.Ready)
		if info.HFSpec.VocabSize > 0 {
			fmt.Fprintf(w, "  Vocab:        %d\n", info.HFSpec.VocabSize)
		}
		if info.HFSpec.ContextLength > 0 {
			fmt.Fprintf(w, "  Context:      %d\n", info.HFSpec.ContextLength)
		}
		if info.HFSpec.EmbeddingLength > 0 {
			fmt.Fprintf(w, "  Embedding:    %d\n", info.HFSpec.EmbeddingLength)
		}
		if info.HFSpec.BlockCount > 0 {
			fmt.Fprintf(w, "  Blocks:       %d\n", info.HFSpec.BlockCount)
		}
		if info.HFSpec.FeedForwardLength > 0 {
			fmt.Fprintf(w, "  FFN:          %d\n", info.HFSpec.FeedForwardLength)
		}
		if info.HFSpec.AttentionHeadCount > 0 {
			fmt.Fprintf(w, "  Heads:        %d\n", info.HFSpec.AttentionHeadCount)
		}
		if info.HFSpec.KVHeadCount > 0 {
			fmt.Fprintf(w, "  KV heads:     %d\n", info.HFSpec.KVHeadCount)
		}
		if info.HFSpec.HeadDim > 0 {
			fmt.Fprintf(w, "  Head dim:     %d\n", info.HFSpec.HeadDim)
		}
		if info.HFSpec.RopeTheta > 0 {
			fmt.Fprintf(w, "  RoPE theta:   %.0f\n", info.HFSpec.RopeTheta)
		}
		if info.HFSpec.RMSNormEpsilon > 0 {
			fmt.Fprintf(w, "  RMS eps:      %g\n", info.HFSpec.RMSNormEpsilon)
		}
		if len(info.HFSpec.Missing) > 0 {
			fmt.Fprintf(w, "  Missing:      %s\n", strings.Join(info.HFSpec.Missing, ", "))
		}
		if info.HFSpec.ValidationError != "" {
			fmt.Fprintf(w, "  Error:        %s\n", info.HFSpec.ValidationError)
		}
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
			fmt.Fprintf(w, "  Tensor bytes: %s\n", humanBytesUint(info.Safetensors.TotalSize))
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
	if info.HFShapes != nil {
		fmt.Fprintln(w, "HF shapes:")
		fmt.Fprintf(w, "  Ready:        %v\n", info.HFShapes.Ready)
		fmt.Fprintf(w, "  Checked:      %d\n", len(info.HFShapes.Checked))
		if len(info.HFShapes.MissingShape) > 0 {
			fmt.Fprintf(w, "  Mismatches:   %d\n", len(info.HFShapes.MissingShape))
			for _, mismatch := range info.HFShapes.MissingShape {
				fmt.Fprintf(w, "    - %s\n", mismatch)
			}
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

func writeGenerationConfig(w io.Writer, cfg *modelinfo.GenerationConfig) {
	if cfg == nil {
		return
	}
	fmt.Fprintln(w, "Generation:")
	if cfg.MaxNewTokens != nil {
		fmt.Fprintf(w, "  Max new:      %d\n", *cfg.MaxNewTokens)
	} else if cfg.MaxLength != nil {
		fmt.Fprintf(w, "  Max length:   %d\n", *cfg.MaxLength)
	}
	if cfg.MinNewTokens != nil {
		fmt.Fprintf(w, "  Min new:      %d\n", *cfg.MinNewTokens)
	}
	if cfg.DoSample != nil {
		fmt.Fprintf(w, "  Sample:       %v\n", *cfg.DoSample)
	}
	if cfg.Temperature != nil {
		fmt.Fprintf(w, "  Temperature:  %g\n", *cfg.Temperature)
	}
	if cfg.TopP != nil {
		fmt.Fprintf(w, "  Top-p:        %g\n", *cfg.TopP)
	}
	if cfg.TopK != nil {
		fmt.Fprintf(w, "  Top-k:        %d\n", *cfg.TopK)
	}
	if cfg.TypicalP != nil {
		fmt.Fprintf(w, "  Typical-p:    %g\n", *cfg.TypicalP)
	}
	if cfg.RepetitionPenalty != nil {
		fmt.Fprintf(w, "  Repeat:       %g\n", *cfg.RepetitionPenalty)
	}
	if cfg.NoRepeatNGramSize != nil {
		fmt.Fprintf(w, "  No repeat n:  %d\n", *cfg.NoRepeatNGramSize)
	}
	if cfg.BOSTokenID != nil {
		fmt.Fprintf(w, "  BOS token:    %d\n", *cfg.BOSTokenID)
	}
	if cfg.PadTokenID != nil {
		fmt.Fprintf(w, "  PAD token:    %d\n", *cfg.PadTokenID)
	}
	if len(cfg.EOSTokenIDs) > 0 {
		fmt.Fprintf(w, "  EOS tokens:   %s\n", joinInts(cfg.EOSTokenIDs))
	}
	if len(cfg.StopStrings) > 0 {
		fmt.Fprintf(w, "  Stop:         %s\n", strings.Join(cfg.StopStrings, ", "))
	}
}

func joinInts(values []int) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.Itoa(value))
	}
	return strings.Join(parts, ", ")
}

func writeNativeAdapterInfo(w io.Writer, info *modelinfo.NativeAdapterInfo) {
	if info == nil {
		return
	}
	fmt.Fprintln(w, "Native adapter:")
	if info.BaseModel != "" {
		fmt.Fprintf(w, "  Base model:   %s\n", info.BaseModel)
	}
	if info.AdapterPath != "" {
		fmt.Fprintf(w, "  Adapter:      %s\n", info.AdapterPath)
	}
	if info.RecommendedBackend != "" {
		fmt.Fprintf(w, "  Backend:      %s\n", info.RecommendedBackend)
	}
	if info.Method != "" {
		fmt.Fprintf(w, "  Method:       %s\n", info.Method)
	}
	if info.DatasetFormat != "" {
		fmt.Fprintf(w, "  Dataset:      %s\n", info.DatasetFormat)
	}
	if info.BaseFormat != "" {
		fmt.Fprintf(w, "  Base format:  %s\n", info.BaseFormat)
	}
	if info.LearningRate > 0 {
		fmt.Fprintf(w, "  Learn rate:   %.6g\n", info.LearningRate)
	}
	if info.Epochs > 0 {
		fmt.Fprintf(w, "  Epochs:       %d\n", info.Epochs)
	}
	if info.MaxContext > 0 {
		fmt.Fprintf(w, "  Max context:  %d\n", info.MaxContext)
	}
	if info.TrainRows > 0 {
		fmt.Fprintf(w, "  Train rows:   %d\n", info.TrainRows)
	}
	if info.EvalRows > 0 {
		fmt.Fprintf(w, "  Eval rows:    %d\n", info.EvalRows)
	}
	if info.DuplicateRows > 0 {
		fmt.Fprintf(w, "  Duplicates:   %d\n", info.DuplicateRows)
	}
	if info.UpdatedTokens > 0 {
		fmt.Fprintf(w, "  Tokens:       %d updated\n", info.UpdatedTokens)
	}
	if info.EvalCoverage != nil {
		printNativeAdapterEvalCoverage(w, info.EvalCoverage)
	}
	printNativeAdapterTopTokens(w, info.TopTokens)
}

func printNativeAdapterEvalCoverage(w io.Writer, summary *modelinfo.NativeAdapterEval) {
	fmt.Fprintf(w, "  Eval coverage: %d/%d tokens (%.2f%%), %d/%d unique (%.2f%%)\n",
		summary.CoveredTokens,
		summary.Tokens,
		summary.Coverage*100,
		summary.CoveredUniqueTokens,
		summary.UniqueTokens,
		summary.UniqueCoverage*100,
	)
}

func printNativeTrainingTopTokens(w io.Writer, tokens []training.NativeTokenSummary) {
	if len(tokens) == 0 {
		return
	}
	fmt.Fprintln(w, "Top tokens:")
	for _, token := range tokens {
		fmt.Fprintf(w, "  - %d", token.ID)
		if token.Text != "" {
			fmt.Fprintf(w, " %s", strconv.Quote(token.Text))
		}
		fmt.Fprintf(w, " count=%d bias=%.4g\n", token.Count, token.Bias)
	}
}

func printNativeAdapterTopTokens(w io.Writer, tokens []modelinfo.NativeAdapterToken) {
	if len(tokens) == 0 {
		return
	}
	fmt.Fprintln(w, "  Top tokens:")
	for _, token := range tokens {
		fmt.Fprintf(w, "    - %d", token.ID)
		if token.Text != "" {
			fmt.Fprintf(w, " %s", strconv.Quote(token.Text))
		}
		fmt.Fprintf(w, " count=%d bias=%.4g\n", token.Count, token.Bias)
	}
}

func printMemoryEstimate(w io.Writer, estimate *modelinfo.MemoryEstimate) {
	fmt.Fprintln(w, "Memory estimate")
	fmt.Fprintf(w, "Path:          %s\n", estimate.Path)
	fmt.Fprintf(w, "Format:        %s\n", estimate.Format)
	if estimate.ContextLength > 0 {
		fmt.Fprintf(w, "Context:       %d\n", estimate.ContextLength)
	}
	if estimate.BlockCount > 0 {
		fmt.Fprintf(w, "Blocks:        %d\n", estimate.BlockCount)
	}
	if estimate.EmbeddingLength > 0 {
		fmt.Fprintf(w, "Embedding:     %d\n", estimate.EmbeddingLength)
	}
	if estimate.AttentionHeadCount > 0 {
		fmt.Fprintf(w, "Heads:         %d\n", estimate.AttentionHeadCount)
	}
	if estimate.KVHeadCount > 0 {
		fmt.Fprintf(w, "KV heads:      %d\n", estimate.KVHeadCount)
	}
	if estimate.HeadDim > 0 {
		fmt.Fprintf(w, "Head dim:      %d\n", estimate.HeadDim)
	}
	if estimate.WeightBytes > 0 {
		fmt.Fprintf(w, "Weights:       %s\n", humanBytesUint(estimate.WeightBytes))
	}
	if estimate.KVCacheBytes > 0 {
		fmt.Fprintf(w, "KV cache:      %s\n", humanBytesUint(estimate.KVCacheBytes))
	}
	if estimate.RuntimeBytes > 0 {
		fmt.Fprintf(w, "Runtime:       %s\n", humanBytesUint(estimate.RuntimeBytes))
	}
	if estimate.TotalBytes > 0 {
		fmt.Fprintf(w, "Total:         %s\n", humanBytesUint(estimate.TotalBytes))
	}
	for _, note := range estimate.Notes {
		fmt.Fprintf(w, "Note:          %s\n", note)
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

func runStatus(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	var runtimeStats bool
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	fs.BoolVar(&runtimeStats, "runtime-stats", false, "load the pure-Go backend and include runtime cache/device stats")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego status <model-path> [flags]")
		return 2
	}
	artifact, err := modelinfo.Resolve(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	var stats nego.RuntimeStats
	if runtimeStats {
		if !jsonOutput {
			fmt.Fprintln(stderr, "Loading runtime for stats...")
		}
		loadedStats, err := loadStatusRuntimeStats(positionals[0])
		if err != nil {
			fmt.Fprintf(stderr, "nego: %v\n", err)
			return 1
		}
		stats = loadedStats
	}
	if jsonOutput {
		if runtimeStats {
			_ = json.NewEncoder(stdout).Encode(struct {
				Artifact     *modelinfo.Artifact `json:"artifact"`
				RuntimeStats nego.RuntimeStats   `json:"runtime_stats"`
			}{
				Artifact:     artifact,
				RuntimeStats: stats,
			})
			return 0
		}
		_ = json.NewEncoder(stdout).Encode(artifact)
		return 0
	}
	writeArtifactStatus(stdout, artifact)
	if runtimeStats {
		writeRuntimeStats(stdout, stats)
	}
	return 0
}

func loadStatusRuntimeStats(path string) (nego.RuntimeStats, error) {
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
		Backend: nego.BackendNativeAuto,
		Path:    path,
		Options: map[string]string{
			"allow_incomplete": "true",
		},
	})
	if err != nil {
		return nego.RuntimeStats{}, fmt.Errorf("load runtime stats: %w", err)
	}
	defer model.Close()
	stats, ok := nego.RuntimeStatsOf(model)
	if !ok {
		return nego.RuntimeStats{}, fmt.Errorf("backend does not expose runtime stats")
	}
	return stats, nil
}

func writeArtifactStatus(w io.Writer, artifact *modelinfo.Artifact) {
	fmt.Fprintf(w, "Path:          %s\n", artifact.Path)
	fmt.Fprintf(w, "Format:        %s\n", artifact.Format)
	fmt.Fprintf(w, "Run:           %s\n", firstNonEmptyString(artifact.RecommendedRunBackend, "not ready"))
	fmt.Fprintf(w, "Train:         %s\n", firstNonEmptyString(artifact.RecommendedTrainBackend, "not ready"))
	if artifact.RuntimeFile != nil {
		fmt.Fprintf(w, "Runtime file:  %s [%s]\n", artifact.RuntimeFile.Path, artifact.RuntimeFile.Kind)
	}
	if artifact.ModelType != "" {
		fmt.Fprintf(w, "Model type:    %s\n", artifact.ModelType)
	}
	if artifact.ContextLength > 0 {
		fmt.Fprintf(w, "Context:       %d\n", artifact.ContextLength)
	}
	if artifact.Quantization != "" {
		fmt.Fprintf(w, "Quantization:  %s\n", artifact.Quantization)
	}
	if artifact.ParameterCount > 0 {
		fmt.Fprintf(w, "Parameters:    %d\n", artifact.ParameterCount)
	}
	fmt.Fprintf(w, "Chat template: %s\n", yesNo(artifact.ChatTemplate))
	writeArtifactCapabilities(w, "Run backends:", artifact.RunBackends)
	writeArtifactCapabilities(w, "Train backends:", artifact.TrainBackends)
	if len(artifact.Warnings) > 0 {
		fmt.Fprintln(w, "Warnings:")
		for _, warning := range artifact.Warnings {
			fmt.Fprintf(w, "  - %s\n", warning)
		}
	}
}

func writeRuntimeStats(w io.Writer, stats nego.RuntimeStats) {
	fmt.Fprintln(w, "Runtime stats:")
	if stats.Backend != "" {
		fmt.Fprintf(w, "  Backend:      %s\n", stats.Backend)
	}
	if stats.Device != "" {
		fmt.Fprintf(w, "  Device:       %s\n", stats.Device)
	}
	fmt.Fprintf(w, "  Cache:        %s\n", enabledDisabled(stats.TensorCacheEnabled))
	fmt.Fprintf(w, "  Tensors:      %d\n", stats.CachedTensors)
	fmt.Fprintf(w, "  Cached bytes: %s\n", humanBytesUint(stats.CachedTensorBytes))
	if stats.MaxTensorCacheBytes > 0 {
		fmt.Fprintf(w, "  Cache limit:  %s\n", humanBytesUint(stats.MaxTensorCacheBytes))
	}
	fmt.Fprintf(w, "  Adapter:      %s\n", yesNo(stats.AdapterLoaded))
	if stats.ExperimentalGeneration {
		fmt.Fprintln(w, "  Generation:   experimental")
	}
}

func enabledDisabled(value bool) string {
	if value {
		return "enabled"
	}
	return "disabled"
}

func writeArtifactCapabilities(w io.Writer, label string, capabilities []modelinfo.ArtifactCapability) {
	if len(capabilities) == 0 {
		return
	}
	fmt.Fprintln(w, label)
	for _, capability := range capabilities {
		fmt.Fprintf(w, "  - %s: %s", capability.Name, capability.Status)
		if capability.Reason != "" {
			fmt.Fprintf(w, " (%s)", capability.Reason)
		}
		fmt.Fprintln(w)
	}
}

func yesNo(value bool) string {
	if value {
		return "yes"
	}
	return "no"
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
	if report.Generation != nil {
		writeGenerationConfig(w, report.Generation)
	}
	if report.ChatTemplate {
		fmt.Fprintln(w, "Chat template: yes")
	} else {
		fmt.Fprintln(w, "Chat template: no")
	}
	if report.Tokenizer != nil {
		writeTokenizerReport(w, report.Tokenizer)
	}
	if report.NativeAdapter != nil {
		writeNativeAdapterInfo(w, report.NativeAdapter)
	}
	if report.NativeReadiness != nil {
		writeNativeReadiness(w, report.NativeReadiness)
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
	if report.Memory != nil {
		writeCheckMemory(w, report.Memory)
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

func writeTokenizerReport(w io.Writer, report *modelinfo.TokenizerReport) {
	fmt.Fprintln(w, "Tokenizer:")
	if report.Format != "" {
		fmt.Fprintf(w, "  Format:       %s\n", report.Format)
	}
	if report.Model != "" {
		fmt.Fprintf(w, "  Model:        %s\n", report.Model)
	}
	if report.PreTokenizer != "" {
		fmt.Fprintf(w, "  Pre-tokenizer: %s\n", report.PreTokenizer)
	}
	if report.VocabSize > 0 {
		fmt.Fprintf(w, "  Vocab size:   %d\n", report.VocabSize)
	}
	if values := tokenizerSpecialIDs(report); len(values) > 0 {
		fmt.Fprintf(w, "  Special IDs:  %s\n", strings.Join(values, ", "))
	}
	if values := tokenizerDefaults(report); len(values) > 0 {
		fmt.Fprintf(w, "  Defaults:     %s\n", strings.Join(values, ", "))
	}
}

func tokenizerSpecialIDs(report *modelinfo.TokenizerReport) []string {
	var out []string
	appendID := func(name string, value *uint32) {
		if value != nil {
			out = append(out, fmt.Sprintf("%s=%d", name, *value))
		}
	}
	appendID("bos", report.BOSTokenID)
	appendID("eos", report.EOSTokenID)
	appendID("unk", report.UNKTokenID)
	appendID("pad", report.PADTokenID)
	appendID("eot", report.EOTTokenID)
	appendID("eom", report.EOMTokenID)
	return out
}

func tokenizerDefaults(report *modelinfo.TokenizerReport) []string {
	var out []string
	appendBool := func(name string, value *bool) {
		if value != nil {
			out = append(out, fmt.Sprintf("%s=%s", name, yesNo(*value)))
		}
	}
	appendBool("add_bos", report.AddBOS)
	appendBool("add_eos", report.AddEOS)
	appendBool("add_space_prefix", report.AddSpacePrefix)
	appendBool("remove_extra_whitespaces", report.RemoveExtraWhitespaces)
	return out
}

func writeCheckMemory(w io.Writer, estimate *modelinfo.MemoryEstimate) {
	fmt.Fprintln(w, "Memory:")
	if estimate.WeightBytes > 0 {
		fmt.Fprintf(w, "  Weights:      %s\n", humanBytesUint(estimate.WeightBytes))
	}
	if estimate.KVCacheBytes > 0 {
		fmt.Fprintf(w, "  KV cache:     %s\n", humanBytesUint(estimate.KVCacheBytes))
	}
	if estimate.RuntimeBytes > 0 {
		fmt.Fprintf(w, "  Runtime:      %s\n", humanBytesUint(estimate.RuntimeBytes))
	}
	if estimate.TotalBytes > 0 {
		fmt.Fprintf(w, "  Total:        %s\n", humanBytesUint(estimate.TotalBytes))
	}
	writeLimitedStringList(w, "  Notes:", estimate.Notes, 2)
}

func writeNativeReadiness(w io.Writer, readiness *modelinfo.NativeRuntimeReadiness) {
	status := "no"
	if readiness.Ready {
		status = "yes"
	}
	fmt.Fprintln(w, "Native readiness:")
	fmt.Fprintf(w, "  Ready:        %s\n", status)
	if readiness.Reason != "" {
		fmt.Fprintf(w, "  Reason:       %s\n", readiness.Reason)
	}
	if readiness.RequiredTensorCount > 0 {
		fmt.Fprintf(w, "  Required:     %d tensors\n", readiness.RequiredTensorCount)
	}
	if readiness.OptionalTensorCount > 0 {
		fmt.Fprintf(w, "  Optional:     %d tensors\n", readiness.OptionalTensorCount)
	}
	writeLimitedStringList(w, "  Spec issues:", readiness.SpecIssues, 6)
	writeLimitedStringList(w, "  Unsupported:", readiness.UnsupportedTensorTypes, 6)
	writeLimitedStringList(w, "  Unsupported required:", readiness.UnsupportedRequired, 6)
	writeLimitedStringList(w, "  Unsupported optional:", readiness.UnsupportedOptional, 6)
	writeLimitedStringList(w, "  Missing:", readiness.MissingTensors, 8)
	writeLimitedStringList(w, "  Shape issues:", readiness.ShapeMismatches, 8)
	writeLimitedStringList(w, "  Warnings:", readiness.Warnings, 4)
}

func writeLimitedStringList(w io.Writer, label string, values []string, limit int) {
	if len(values) == 0 {
		return
	}
	if limit <= 0 || limit > len(values) {
		limit = len(values)
	}
	fmt.Fprintln(w, label)
	for _, value := range values[:limit] {
		fmt.Fprintf(w, "    - %s\n", value)
	}
	if remaining := len(values) - limit; remaining > 0 {
		fmt.Fprintf(w, "    - ... %d more\n", remaining)
	}
}

func runModel(args []string, stdout, stderr io.Writer) int {
	var backend string
	var maxTokens int
	var temperature float64
	var topK int
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
	var noCacheTensors bool
	var maxTensorCacheBytes uint64
	var native bool
	var adapterPath string
	var configFile string
	var logPath string
	var extraOptions repeatedFlag
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "auto", "runtime backend: auto, native, native-hf, llama.cpp, or openai-compatible")
	fs.BoolVar(&native, "native", false, "use the experimental pure-Go native backend")
	fs.IntVar(&maxTokens, "max-tokens", 0, "maximum tokens to generate")
	fs.Float64Var(&temperature, "temperature", 0, "sampling temperature")
	fs.IntVar(&topK, "top-k", 0, "keep only the top k tokens during sampling")
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
	fs.BoolVar(&noCacheTensors, "no-cache-tensors", false, "disable native decoded tensor cache")
	fs.Uint64Var(&maxTensorCacheBytes, "max-tensor-cache-bytes", 0, "maximum native decoded tensor cache bytes; 0 means unlimited")
	fs.StringVar(&adapterPath, "adapter", "", "native adapter JSON")
	fs.StringVar(&configFile, "f", "", "run config file")
	fs.StringVar(&logPath, "log", "", "append run result to JSONL log")
	fs.Var(&extraOptions, "option", "backend option key=value, repeatable")
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
	if cfg.MaxTokens > 0 && maxTokens == 0 {
		maxTokens = cfg.MaxTokens
	}
	if cfg.Temperature > 0 && temperature == 0 {
		temperature = cfg.Temperature
	}
	if cfg.TopK != 0 && topK == 0 {
		topK = cfg.TopK
	}
	if cfg.TopP > 0 && topP == 0 {
		topP = cfg.TopP
	}
	if cfg.RepeatPenalty > 0 && repeatPenalty == 0 {
		repeatPenalty = cfg.RepeatPenalty
	}
	if err := validateTopK(topK); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
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
	defaults := generationDefaultOverrides{
		maxTokens:     flagWasSet(fs, "max-tokens") || cfg.MaxTokens > 0,
		temperature:   flagWasSet(fs, "temperature") || cfg.Temperature > 0,
		topK:          flagWasSet(fs, "top-k") || cfg.TopK > 0,
		topP:          flagWasSet(fs, "top-p") || cfg.TopP > 0,
		repeatPenalty: flagWasSet(fs, "repeat-penalty") || cfg.RepeatPenalty > 0,
		stop:          flagWasSet(fs, "stop") || len(cfg.Stop) > 0,
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
	if native {
		backend, err = resolvePureGoBackend(path)
		if err != nil {
			fmt.Fprintf(stderr, "nego: %v\n", err)
			return 1
		}
	}
	if err := applyCLIGenerationDefaults(path, defaults, &maxTokens, &temperature, &topK, &topP, &repeatPenalty, &stop); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if err := validateTopK(topK); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if err := validateRepeatPenalty(repeatPenalty); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	options, err := runtimeOptions(cfg.Options, runtimeFlagOptions{
		threads:             threads,
		ctxSize:             ctxSize,
		gpuLayers:           gpuLayers,
		gpuMode:             gpuMode,
		mainGPU:             mainGPU,
		tensorSplit:         tensorSplit,
		splitMode:           splitMode,
		flashAttention:      flashAttention,
		noCacheTensors:      noCacheTensors,
		maxTensorCacheBytes: maxTensorCacheBytes,
		adapterPath:         adapterPath,
		extraOptions:        extraOptions,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if err := validateRuntimeOptions(options); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	started := time.Now().UTC()
	backend = resolveRuntimeBackend(backend, path, cfg.Endpoint, cfg.Model)
	printModelLoadProgress(stderr, backend, path)
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{Backend: backend, Path: path, Endpoint: cfg.Endpoint, Model: cfg.Model, APIKey: cfg.APIKey, Options: options})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("run", backend, path, cfg.Endpoint, cfg.Model, prompt, nil, "", started, maxTokens, temperature, topK, topP, repeatPenalty, stop, seed, options, err)); logErr != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
		}
		return 1
	}
	defer model.Close()
	out, err := model.Generate(context.Background(), nego.GenerateRequest{Prompt: prompt, MaxTokens: maxTokens, Temperature: temperature, TopK: topK, TopP: topP, RepeatPenalty: repeatPenalty, Stop: stop, Seed: seed})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("run", backend, path, cfg.Endpoint, cfg.Model, prompt, nil, "", started, maxTokens, temperature, topK, topP, repeatPenalty, stop, seed, options, err, runtimeStatsForLog(model))); logErr != nil {
			fmt.Fprintf(stderr, "nego: write run log: %v\n", logErr)
		}
		return 1
	}
	fmt.Fprint(stdout, out.Text)
	if err := appendRuntimeLog(logPath, runtimeLogEntry("run", backend, path, cfg.Endpoint, cfg.Model, prompt, nil, out.Text, started, maxTokens, temperature, topK, topP, repeatPenalty, stop, seed, options, nil, runtimeStatsForLog(model))); err != nil {
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
	var topK int
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
	var noCacheTensors bool
	var maxTensorCacheBytes uint64
	var configFile string
	var logPath string
	var sessionPath string
	var savePath string
	var interactive bool
	var native bool
	var adapterPath string
	var extraOptions repeatedFlag
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "auto", "runtime backend: auto, native, llama.cpp, or openai-compatible")
	fs.BoolVar(&native, "native", false, "use the experimental pure-Go native backend")
	fs.StringVar(&system, "system", "", "system message")
	fs.IntVar(&maxTokens, "max-tokens", 0, "maximum tokens to generate")
	fs.Float64Var(&temperature, "temperature", 0, "sampling temperature")
	fs.IntVar(&topK, "top-k", 0, "keep only the top k tokens during sampling")
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
	fs.BoolVar(&noCacheTensors, "no-cache-tensors", false, "disable native decoded tensor cache")
	fs.Uint64Var(&maxTensorCacheBytes, "max-tensor-cache-bytes", 0, "maximum native decoded tensor cache bytes; 0 means unlimited")
	fs.StringVar(&adapterPath, "adapter", "", "native adapter JSON")
	fs.StringVar(&configFile, "f", "", "chat config file")
	fs.StringVar(&logPath, "log", "", "append run result to JSONL log")
	fs.StringVar(&sessionPath, "session", "", "load and save chat history JSON")
	fs.StringVar(&savePath, "save", "", "save chat history JSON without loading it")
	fs.BoolVar(&interactive, "interactive", false, "start an interactive chat session")
	fs.Var(&extraOptions, "option", "backend option key=value, repeatable")
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
	if cfg.TopK != 0 && topK == 0 {
		topK = cfg.TopK
	}
	if cfg.TopP > 0 && topP == 0 {
		topP = cfg.TopP
	}
	if cfg.RepeatPenalty > 0 && repeatPenalty == 0 {
		repeatPenalty = cfg.RepeatPenalty
	}
	if err := validateTopK(topK); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
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
	defaults := generationDefaultOverrides{
		maxTokens:     flagWasSet(fs, "max-tokens") || cfg.MaxTokens > 0,
		temperature:   flagWasSet(fs, "temperature") || cfg.Temperature > 0,
		topK:          flagWasSet(fs, "top-k") || cfg.TopK > 0,
		topP:          flagWasSet(fs, "top-p") || cfg.TopP > 0,
		repeatPenalty: flagWasSet(fs, "repeat-penalty") || cfg.RepeatPenalty > 0,
		stop:          flagWasSet(fs, "stop") || len(cfg.Stop) > 0,
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
	if native {
		backend, err = resolvePureGoBackend(path)
		if err != nil {
			fmt.Fprintf(stderr, "nego: %v\n", err)
			return 1
		}
	}
	if err := applyCLIGenerationDefaults(path, defaults, &maxTokens, &temperature, &topK, &topP, &repeatPenalty, &stop); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if err := validateTopK(topK); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if err := validateRepeatPenalty(repeatPenalty); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	options, err := runtimeOptions(cfg.Options, runtimeFlagOptions{
		threads:             threads,
		ctxSize:             ctxSize,
		gpuLayers:           gpuLayers,
		gpuMode:             gpuMode,
		mainGPU:             mainGPU,
		tensorSplit:         tensorSplit,
		splitMode:           splitMode,
		flashAttention:      flashAttention,
		noCacheTensors:      noCacheTensors,
		maxTensorCacheBytes: maxTensorCacheBytes,
		adapterPath:         adapterPath,
		extraOptions:        extraOptions,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if err := validateRuntimeOptions(options); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	backend = resolveRuntimeBackend(backend, path, cfg.Endpoint, cfg.Model)
	printModelLoadProgress(stderr, backend, path)
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{Backend: backend, Path: path, Endpoint: cfg.Endpoint, Model: cfg.Model, APIKey: cfg.APIKey, Options: options})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		started := time.Now().UTC()
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("chat", backend, path, cfg.Endpoint, cfg.Model, "", messages, "", started, maxTokens, temperature, topK, topP, repeatPenalty, stop, seed, options, err)); logErr != nil {
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
			topK:          topK,
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
	resp, err := model.Chat(context.Background(), nego.ChatRequest{Messages: messages, MaxTokens: maxTokens, Temperature: temperature, TopK: topK, TopP: topP, RepeatPenalty: repeatPenalty, Stop: stop, Seed: seed})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		if logErr := appendRuntimeLog(logPath, runtimeLogEntry("chat", backend, path, cfg.Endpoint, cfg.Model, "", messages, "", started, maxTokens, temperature, topK, topP, repeatPenalty, stop, seed, options, err, runtimeStatsForLog(model))); logErr != nil {
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
	if err := appendRuntimeLog(logPath, runtimeLogEntry("chat", backend, path, cfg.Endpoint, cfg.Model, "", messages, resp.Message.Content, started, maxTokens, temperature, topK, topP, repeatPenalty, stop, seed, options, nil, runtimeStatsForLog(model))); err != nil {
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
	topK          int
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
			TopK:          opts.topK,
			TopP:          opts.topP,
			RepeatPenalty: opts.repeatPenalty,
			Stop:          opts.stop,
			Seed:          opts.seed,
		})
		if err != nil {
			fmt.Fprintln(stdout)
			fmt.Fprintf(stderr, "nego: %v\n", err)
			if logErr := appendRuntimeLog(opts.logPath, runtimeLogEntry("chat", opts.backend, opts.path, opts.endpoint, opts.modelID, "", messages, "", started, opts.maxTokens, opts.temperature, opts.topK, opts.topP, opts.repeatPenalty, opts.stop, opts.seed, opts.options, err, runtimeStatsForLog(model))); logErr != nil {
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
			if logErr := appendRuntimeLog(opts.logPath, runtimeLogEntry("chat", opts.backend, opts.path, opts.endpoint, opts.modelID, "", messages, output.String(), started, opts.maxTokens, opts.temperature, opts.topK, opts.topP, opts.repeatPenalty, opts.stop, opts.seed, opts.options, err, runtimeStatsForLog(model))); logErr != nil {
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
		if err := appendRuntimeLog(opts.logPath, runtimeLogEntry("chat", opts.backend, opts.path, opts.endpoint, opts.modelID, "", messages, reply, started, opts.maxTokens, opts.temperature, opts.topK, opts.topP, opts.repeatPenalty, opts.stop, opts.seed, opts.options, nil, runtimeStatsForLog(model))); err != nil {
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
	TopK          int               `json:"top_k"`
	TopP          float64           `json:"top_p"`
	RepeatPenalty float64           `json:"repeat_penalty"`
	Stop          []string          `json:"stop"`
	Seed          int64             `json:"seed"`
	Log           string            `json:"log"`
	Session       string            `json:"session"`
	Save          string            `json:"save"`
}

type runtimeConfigOverrides struct {
	prompt        string
	system        string
	logPath       string
	maxTokens     int
	temperature   float64
	topK          int
	topP          float64
	repeatPenalty float64
	seed          int64
}

func buildRuntimeConfig(path string, overrides runtimeConfigOverrides) (runtimeConfig, error) {
	artifact, err := modelinfo.Resolve(path)
	if err != nil {
		return runtimeConfig{}, err
	}
	if artifact.RecommendedRunBackend == "" {
		return runtimeConfig{}, modelinfo.FormatResolveError(path, artifact)
	}
	cfg := runtimeConfig{
		Backend:       artifact.RecommendedRunBackend,
		Path:          path,
		Prompt:        overrides.prompt,
		System:        overrides.system,
		Log:           overrides.logPath,
		MaxTokens:     overrides.maxTokens,
		Temperature:   overrides.temperature,
		TopK:          overrides.topK,
		TopP:          overrides.topP,
		RepeatPenalty: overrides.repeatPenalty,
		Seed:          overrides.seed,
	}
	applyRuntimeConfigGenerationDefaults(&cfg, path, artifact)
	return cfg, nil
}

func applyRuntimeConfigGenerationDefaults(cfg *runtimeConfig, path string, artifact *modelinfo.Artifact) {
	paths := []string{path}
	if artifact != nil && artifact.NativeAdapter != nil && artifact.NativeAdapter.BaseModel != "" {
		paths = append(paths, artifact.NativeAdapter.BaseModel)
	}
	for _, candidate := range paths {
		gen, err := modelinfo.LoadGenerationConfig(candidate)
		if err != nil || gen == nil {
			continue
		}
		if cfg.MaxTokens == 0 && gen.MaxNewTokens != nil {
			cfg.MaxTokens = *gen.MaxNewTokens
		}
		if cfg.Temperature == 0 && gen.Temperature != nil {
			cfg.Temperature = *gen.Temperature
		}
		if cfg.TopK == 0 && gen.TopK != nil {
			cfg.TopK = *gen.TopK
		}
		if cfg.TopP == 0 && gen.TopP != nil {
			cfg.TopP = *gen.TopP
		}
		if cfg.RepeatPenalty == 0 && gen.RepetitionPenalty != nil {
			cfg.RepeatPenalty = *gen.RepetitionPenalty
		}
		if len(cfg.Stop) == 0 && len(gen.StopStrings) > 0 {
			cfg.Stop = append([]string(nil), gen.StopStrings...)
		}
		return
	}
}

func writeRuntimeConfig(path string, cfg runtimeConfig) error {
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
	return encoder.Encode(cfg)
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

type generationDefaultOverrides struct {
	maxTokens     bool
	temperature   bool
	topK          bool
	topP          bool
	repeatPenalty bool
	stop          bool
}

func applyCLIGenerationDefaults(path string, overrides generationDefaultOverrides, maxTokens *int, temperature *float64, topK *int, topP, repeatPenalty *float64, stop *repeatedFlag) error {
	if path == "" {
		return nil
	}
	cfg, err := modelinfo.LoadGenerationConfig(path)
	if err != nil || cfg == nil {
		return err
	}
	if !overrides.maxTokens && maxTokens != nil && *maxTokens == 0 && cfg.MaxNewTokens != nil {
		*maxTokens = *cfg.MaxNewTokens
	}
	if !overrides.temperature && temperature != nil && *temperature == 0 && cfg.Temperature != nil {
		*temperature = *cfg.Temperature
	}
	if !overrides.topK && topK != nil && *topK == 0 && cfg.TopK != nil {
		*topK = *cfg.TopK
	}
	if !overrides.topP && topP != nil && *topP == 0 && cfg.TopP != nil {
		*topP = *cfg.TopP
	}
	if !overrides.repeatPenalty && repeatPenalty != nil && *repeatPenalty == 0 && cfg.RepetitionPenalty != nil {
		*repeatPenalty = *cfg.RepetitionPenalty
	}
	if !overrides.stop && stop != nil && len(*stop) == 0 && len(cfg.StopStrings) > 0 {
		*stop = append((*stop)[:0], cfg.StopStrings...)
	}
	return nil
}

func appendRuntimeLog(path string, entry runs.Entry) error {
	if path == "" {
		return nil
	}
	return runs.Append(path, entry)
}

func runtimeLogEntry(command, backend, path, endpoint, modelID, prompt string, messages []nego.Message, output string, started time.Time, maxTokens int, temperature float64, topK int, topP, repeatPenalty float64, stop []string, seed int64, options map[string]string, runErr error, runtimeStats ...*nego.RuntimeStats) runs.Entry {
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
		TopK:          topK,
		TopP:          topP,
		RepeatPenalty: repeatPenalty,
		Stop:          append([]string(nil), stop...),
		Seed:          seed,
		Options:       redactRuntimeOptions(options),
	}
	if runErr != nil {
		entry.Error = runErr.Error()
	}
	if len(runtimeStats) > 0 && runtimeStats[0] != nil {
		stats := *runtimeStats[0]
		entry.Runtime = &stats
	}
	return entry
}

func runtimeStatsForLog(model nego.Model) *nego.RuntimeStats {
	stats, ok := nego.RuntimeStatsOf(model)
	if !ok {
		return nil
	}
	return &stats
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

func redactRuntimeOptions(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		if sensitiveRuntimeOption(key) {
			out[key] = "<redacted>"
			continue
		}
		out[key] = value
	}
	return out
}

func sensitiveRuntimeOption(key string) bool {
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

type runtimeFlagOptions struct {
	threads             int
	ctxSize             int
	gpuLayers           int
	gpuMode             string
	mainGPU             int
	tensorSplit         string
	splitMode           string
	flashAttention      bool
	noCacheTensors      bool
	maxTensorCacheBytes uint64
	adapterPath         string
	extraOptions        []string
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

func resolvePureGoBackend(path string) (string, error) {
	return nego.ResolvePureGoBackend(nego.ModelOptions{Path: path})
}

func runtimeOptions(base map[string]string, flags runtimeFlagOptions) (map[string]string, error) {
	options := make(map[string]string, len(base)+len(flags.extraOptions)+9)
	for key, value := range base {
		if value != "" {
			options[key] = value
		}
	}
	for _, raw := range flags.extraOptions {
		key, value, err := parseRuntimeOption(raw)
		if err != nil {
			return nil, err
		}
		options[key] = value
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
	if flags.noCacheTensors {
		options["cache_tensors"] = "false"
	}
	if flags.maxTensorCacheBytes > 0 {
		options["max_tensor_cache_bytes"] = strconv.FormatUint(flags.maxTensorCacheBytes, 10)
	}
	if flags.adapterPath != "" {
		options["adapter_path"] = flags.adapterPath
	}
	if len(options) == 0 {
		return nil, nil
	}
	return options, nil
}

func printModelLoadProgress(w io.Writer, backend, path string) {
	label := strings.TrimSpace(backend)
	if label == "" {
		label = "auto"
	}
	fmt.Fprintf(w, "Loading model (%s): %s\n", label, path)
}

func parseRuntimeOption(raw string) (string, string, error) {
	key, value, ok := strings.Cut(raw, "=")
	key = strings.TrimSpace(key)
	value = strings.TrimSpace(value)
	if !ok || key == "" {
		return "", "", fmt.Errorf("option must be key=value")
	}
	if strings.ContainsAny(key, " \t\r\n") {
		return "", "", fmt.Errorf("option key %q must not contain whitespace", key)
	}
	return key, value, nil
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

func validateTopK(value int) error {
	if value < 0 {
		return fmt.Errorf("top-k must be greater than or equal to 0")
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
	for _, key := range []string{"max_tensor_read_bytes", "max_tensor_cache_bytes"} {
		if value := options[key]; value != "" {
			if _, err := strconv.ParseUint(value, 10, 64); err != nil {
				return fmt.Errorf("%s must be an unsigned integer", key)
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
	var noCacheTensors bool
	var maxTensorCacheBytes uint64
	var native bool
	var adapterPath string
	var extraOptions repeatedFlag

	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "auto", "runtime backend: auto, native, native-hf, llama.cpp, or openai-compatible")
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
	fs.BoolVar(&noCacheTensors, "no-cache-tensors", false, "disable native decoded tensor cache")
	fs.Uint64Var(&maxTensorCacheBytes, "max-tensor-cache-bytes", 0, "maximum native decoded tensor cache bytes; 0 means unlimited")
	fs.StringVar(&adapterPath, "adapter", "", "native adapter JSON")
	fs.Var(&extraOptions, "option", "backend option key=value, repeatable")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if native && flagWasSet(fs, "backend") {
		fmt.Fprintln(stderr, "nego: use either --native or --backend, not both")
		return 2
	}
	backend = normalizeBackendFlag(backend)
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego serve <model-path> [flags]")
		return 2
	}
	if native {
		var err error
		backend, err = resolvePureGoBackend(positionals[0])
		if err != nil {
			fmt.Fprintf(stderr, "nego: %v\n", err)
			return 1
		}
	}
	options, err := runtimeOptions(nil, runtimeFlagOptions{
		threads:             threads,
		ctxSize:             ctxSize,
		gpuLayers:           gpuLayers,
		gpuMode:             gpuMode,
		mainGPU:             mainGPU,
		tensorSplit:         tensorSplit,
		splitMode:           splitMode,
		flashAttention:      flashAttention,
		noCacheTensors:      noCacheTensors,
		maxTensorCacheBytes: maxTensorCacheBytes,
		adapterPath:         adapterPath,
		extraOptions:        extraOptions,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	if err := validateRuntimeOptions(options); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 2
	}
	generation, err := modelinfo.LoadGenerationConfig(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	backend = resolveRuntimeBackend(backend, positionals[0], "", "")
	printModelLoadProgress(stderr, backend, positionals[0])
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
	handler, err := server.NewHandler(server.HandlerOptions{ModelID: modelID, Model: model, Generation: generation})
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
	case "compare":
		return runRunsCompare(args[1:], stdout, stderr)
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
	case "quality":
		return runDatasetQuality(args[1:], stdout, stderr)
	case "validate":
		return runDatasetValidate(args[1:], stdout, stderr)
	case "tokens":
		return runDatasetTokens(args[1:], stdout, stderr)
	case "render":
		return runDatasetRender(args[1:], stdout, stderr)
	case "convert":
		return runDatasetConvert(args[1:], stdout, stderr)
	case "filter":
		return runDatasetFilter(args[1:], stdout, stderr)
	case "dedupe":
		return runDatasetDedupe(args[1:], stdout, stderr)
	case "shuffle":
		return runDatasetShuffle(args[1:], stdout, stderr)
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

func runDatasetQuality(args []string, stdout, stderr io.Writer) int {
	var format string
	var requireFields string
	var keys repeatedFlag
	var trimSpace bool
	var ignoreCase bool
	var jsonOutput bool
	fs := flag.NewFlagSet("dataset quality", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&format, "format", "auto", "dataset format: auto, chat, completion, or instruction")
	fs.StringVar(&requireFields, "require", "", "comma-separated fields that must be present and non-empty")
	fs.Var(&keys, "key", "field used to count duplicates, repeatable")
	fs.BoolVar(&trimSpace, "trim-space", false, "trim string fields before duplicate checks")
	fs.BoolVar(&ignoreCase, "ignore-case", false, "case-fold string fields before duplicate checks")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego dataset quality <file> [flags]")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	report := datasets.AnalyzeQuality(rows, datasets.QualityOptions{
		Format:     format,
		Required:   splitCommaFields(requireFields),
		DedupeKeys: expandRepeatedCommaFields(keys),
		TrimSpace:  trimSpace,
		IgnoreCase: ignoreCase,
	})
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(report)
	} else {
		writeDatasetQuality(stdout, report)
	}
	if !report.Valid {
		return 1
	}
	return 0
}

func writeDatasetQuality(w io.Writer, report datasets.QualityReport) {
	status := "ok"
	if !report.Valid {
		status = "invalid"
	}
	fmt.Fprintln(w, "Dataset quality")
	fmt.Fprintf(w, "Rows:       %d\n", report.Rows)
	fmt.Fprintf(w, "Format:     %s\n", report.Format)
	fmt.Fprintf(w, "Status:     %s\n", status)
	if report.Duplicates > 0 {
		fmt.Fprintf(w, "Duplicates: %d\n", report.Duplicates)
	}
	if report.Error != "" {
		fmt.Fprintf(w, "Error:      %s\n", report.Error)
	}
	if len(report.Fields) > 0 {
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "FIELD\tPRESENT\tMISSING\tEMPTY")
		for _, field := range report.Fields {
			fmt.Fprintf(tw, "%s\t%d\t%d\t%d\n", field.Field, field.Present, field.Missing, field.Empty)
		}
		_ = tw.Flush()
	}
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

func runDatasetTokens(args []string, stdout, stderr io.Writer) int {
	var modelPath string
	var format string
	var maxContext int
	var jsonOutput bool
	var top int
	fs := flag.NewFlagSet("dataset tokens", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&modelPath, "model", "", "model path containing tokenizer.json or GGUF vocabulary metadata")
	fs.StringVar(&format, "format", "auto", "dataset format: auto, chat, completion, or instruction")
	fs.IntVar(&maxContext, "max-context", 0, "maximum allowed context tokens")
	fs.IntVar(&top, "top", 5, "number of longest rows to show")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || modelPath == "" {
		fmt.Fprintln(stderr, "usage: nego dataset tokens <file> --model <model-path> [flags]")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	tok, err := loadCLITokenizer(modelPath)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	report, err := datasets.AnalyzeTokenBudgetWithTokenizer(rows, datasets.TokenBudgetOptions{
		ModelPath:  modelPath,
		Format:     format,
		MaxContext: maxContext,
	}, tok)
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(report)
	} else {
		printDatasetTokenBudget(stdout, positionals[0], modelPath, report, top)
	}
	if !report.Valid {
		return 1
	}
	return 0
}

func runDatasetRender(args []string, stdout, stderr io.Writer) int {
	var modelPath string
	var format string
	var n int
	var jsonOutput bool
	fs := flag.NewFlagSet("dataset render", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&modelPath, "model", "", "model directory for chat template rendering")
	fs.StringVar(&format, "format", "auto", "dataset format: auto, chat, completion, or instruction")
	fs.IntVar(&n, "n", 3, "number of rows to render, 0 for all")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego dataset render <file> [flags]")
		return 2
	}
	if n < 0 {
		fmt.Fprintln(stderr, "nego: n must be greater than or equal to 0")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if n > 0 && n < len(rows) {
		rows = rows[:n]
	}
	rendered, err := datasets.RenderTrainingRows(rows, datasets.RenderOptions{
		ModelPath: modelPath,
		Format:    format,
	})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(rendered)
		return 0
	}
	printRenderedDatasetRows(stdout, positionals[0], modelPath, rendered)
	return 0
}

func printRenderedDatasetRows(w io.Writer, path, modelPath string, rows []datasets.RenderedTrainingText) {
	fmt.Fprintln(w, "Rendered dataset")
	fmt.Fprintf(w, "Path:   %s\n", path)
	if modelPath != "" {
		fmt.Fprintf(w, "Model:  %s\n", modelPath)
	}
	fmt.Fprintf(w, "Rows:   %d\n", len(rows))
	for _, row := range rows {
		fmt.Fprintf(w, "\nRow %d [%s]\n", row.Row, row.Format)
		fmt.Fprintln(w, "---")
		fmt.Fprintln(w, row.Text)
		fmt.Fprintln(w, "---")
	}
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

func runDatasetDedupe(args []string, stdout, stderr io.Writer) int {
	var output string
	var keys repeatedFlag
	var trimSpace bool
	var ignoreCase bool
	var jsonOutput bool
	fs := flag.NewFlagSet("dataset dedupe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&output, "out", "", "output JSONL file, or - for stdout")
	fs.Var(&keys, "key", "field used to detect duplicates, repeatable")
	fs.BoolVar(&trimSpace, "trim-space", false, "trim string fields before comparing")
	fs.BoolVar(&ignoreCase, "ignore-case", false, "case-fold string fields before comparing")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON summary")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || output == "" {
		fmt.Fprintln(stderr, "usage: nego dataset dedupe <file> --out <file> [flags]")
		return 2
	}
	if output == "-" && jsonOutput {
		fmt.Fprintln(stderr, "nego: --json cannot be used with --out -")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	result := datasets.DedupeRows(rows, datasets.DedupeOptions{
		Keys:       expandRepeatedCommaFields(keys),
		TrimSpace:  trimSpace,
		IgnoreCase: ignoreCase,
	})
	if err := writeDatasetRows(output, result.Rows, stdout); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(map[string]any{
			"input_rows":  len(rows),
			"output_rows": len(result.Rows),
			"removed":     result.Removed,
			"keys":        expandRepeatedCommaFields(keys),
			"output":      output,
		})
		return 0
	}
	if output != "-" {
		fmt.Fprintf(stdout, "Deduped: %d rows -> %d rows (removed %d) -> %s\n", len(rows), len(result.Rows), result.Removed, output)
	}
	return 0
}

func runDatasetShuffle(args []string, stdout, stderr io.Writer) int {
	var output string
	var seed int64
	fs := flag.NewFlagSet("dataset shuffle", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&output, "out", "", "output JSONL file, or - for stdout")
	fs.Int64Var(&seed, "seed", 42, "shuffle seed")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 || output == "" {
		fmt.Fprintln(stderr, "usage: nego dataset shuffle <file> --out <file> [flags]")
		return 2
	}
	rows, err := datasets.ReadFile(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	shuffled := datasets.Shuffle(rows, seed)
	if err := writeDatasetRows(output, shuffled, stdout); err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	if output != "-" {
		fmt.Fprintf(stdout, "Shuffled: %d rows -> %s\n", len(shuffled), output)
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
	fmt.Fprintln(w, "  nego dataset quality <file> [flags]")
	fmt.Fprintln(w, "  nego dataset validate <file> [flags]")
	fmt.Fprintln(w, "  nego dataset tokens <file> --model <model-path> [flags]")
	fmt.Fprintln(w, "  nego dataset render <file> [flags]")
	fmt.Fprintln(w, "  nego dataset convert <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset filter <file> --where <expr> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset dedupe <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset shuffle <file> --out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset split <file> --train-out <file> --test-out <file> [flags]")
	fmt.Fprintln(w, "  nego dataset sample <file> [flags]")
}

func printDatasetTokenBudget(w io.Writer, path, modelPath string, report datasets.TokenBudgetReport, top int) {
	fmt.Fprintln(w, "Dataset token budget")
	fmt.Fprintf(w, "Path:         %s\n", path)
	fmt.Fprintf(w, "Model:        %s\n", modelPath)
	fmt.Fprintf(w, "Format:       %s\n", report.Format)
	fmt.Fprintf(w, "Rows:         %d\n", report.Rows)
	fmt.Fprintf(w, "Counted rows: %d\n", report.CountedRows)
	fmt.Fprintf(w, "Tokens:       min %d / avg %.1f / max %d / total %d\n", report.MinTokens, report.AverageTokens, report.MaxTokens, report.TotalTokens)
	if report.MaxContext > 0 {
		fmt.Fprintf(w, "Max context:  %d\n", report.MaxContext)
		fmt.Fprintf(w, "Over limit:   %d\n", report.OverLimit)
	}
	var invalid []datasets.TokenBudgetRow
	for _, row := range report.RowsDetail {
		if row.Error != "" {
			invalid = append(invalid, row)
		}
	}
	if len(invalid) > 0 {
		fmt.Fprintln(w, "Invalid rows:")
		for _, row := range invalid {
			fmt.Fprintf(w, "  - row %d: %s\n", row.Row, row.Error)
		}
	}
	longest := datasets.LongestTokenRows(report.RowsDetail, top)
	if len(longest) > 0 {
		fmt.Fprintln(w, "Longest rows:")
		for _, row := range longest {
			status := "ok"
			if row.Error != "" {
				status = "invalid"
			} else if row.OverLimit {
				status = "over limit"
			}
			fmt.Fprintf(w, "  - row %d: %d tokens (%s)\n", row.Row, row.Tokens, status)
		}
	}
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

func expandRepeatedCommaFields(values []string) []string {
	var fields []string
	for _, value := range values {
		fields = append(fields, splitCommaFields(value)...)
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
		target := entry.Model
		if target == "" {
			target = entry.Path
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%dms\n", entry.ID, entry.StartedAt.Format(time.RFC3339), entry.Command, entry.Backend, target, runEntryStatus(entry), entry.DurationMS)
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

func runRunsCompare(args []string, stdout, stderr io.Writer) int {
	var jsonOutput bool
	fs := flag.NewFlagSet("runs compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 3 {
		fmt.Fprintln(stderr, "usage: nego runs compare <runs.jsonl> <baseline-id> <candidate-id> [flags]")
		return 2
	}
	entries, err := runs.Read(positionals[0])
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	baseline, ok := runs.Find(entries, positionals[1])
	if !ok {
		fmt.Fprintf(stderr, "nego: run %q not found\n", positionals[1])
		return 4
	}
	candidate, ok := runs.Find(entries, positionals[2])
	if !ok {
		fmt.Fprintf(stderr, "nego: run %q not found\n", positionals[2])
		return 4
	}
	comparison := compareRuns(baseline, candidate)
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(comparison)
		return 0
	}
	writeRunsComparison(stdout, comparison)
	return 0
}

func runsUsage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  nego runs list <runs.jsonl> [flags]")
	fmt.Fprintln(w, "  nego runs show <runs.jsonl> <id> [flags]")
	fmt.Fprintln(w, "  nego runs compare <runs.jsonl> <baseline-id> <candidate-id> [flags]")
}

type runComparison struct {
	Baseline  runComparisonItem  `json:"baseline"`
	Candidate runComparisonItem  `json:"candidate"`
	Delta     runComparisonDelta `json:"delta"`
}

type runComparisonItem struct {
	ID           string                     `json:"id"`
	Command      string                     `json:"command"`
	Backend      string                     `json:"backend,omitempty"`
	Target       string                     `json:"target,omitempty"`
	Status       string                     `json:"status"`
	DurationMS   int64                      `json:"duration_ms"`
	OutputChars  int                        `json:"output_chars,omitempty"`
	TrainRows    int                        `json:"train_rows,omitempty"`
	EvalRows     int                        `json:"eval_rows,omitempty"`
	Runtime      *runComparisonRuntime      `json:"runtime,omitempty"`
	EvalCoverage *runComparisonEvalCoverage `json:"eval_coverage,omitempty"`
}

type runComparisonDelta struct {
	DurationMS               int64   `json:"duration_ms"`
	OutputChars              int     `json:"output_chars,omitempty"`
	TrainRows                int     `json:"train_rows,omitempty"`
	EvalRows                 int     `json:"eval_rows,omitempty"`
	CachedTensors            int     `json:"cached_tensors,omitempty"`
	CachedTensorBytes        int64   `json:"cached_tensor_bytes,omitempty"`
	EvalCoveragePercentage   float64 `json:"eval_coverage_percentage,omitempty"`
	UniqueCoveragePercentage float64 `json:"unique_coverage_percentage,omitempty"`
}

type runComparisonRuntime struct {
	Backend             string `json:"backend,omitempty"`
	Device              string `json:"device,omitempty"`
	TensorCacheEnabled  bool   `json:"tensor_cache_enabled"`
	CachedTensors       int    `json:"cached_tensors"`
	CachedTensorBytes   uint64 `json:"cached_tensor_bytes"`
	MaxTensorCacheBytes uint64 `json:"max_tensor_cache_bytes,omitempty"`
	AdapterLoaded       bool   `json:"adapter_loaded,omitempty"`
}

type runComparisonEvalCoverage struct {
	Rows                     int     `json:"rows"`
	Tokens                   int     `json:"tokens"`
	CoveredTokens            int     `json:"covered_tokens"`
	Coverage                 float64 `json:"coverage"`
	CoveragePercentage       float64 `json:"coverage_percentage"`
	UniqueTokens             int     `json:"unique_tokens"`
	CoveredUniqueTokens      int     `json:"covered_unique_tokens"`
	UniqueCoverage           float64 `json:"unique_coverage"`
	UniqueCoveragePercentage float64 `json:"unique_coverage_percentage"`
}

func compareRuns(baseline, candidate runs.Entry) runComparison {
	base := runComparisonItemFromEntry(baseline)
	next := runComparisonItemFromEntry(candidate)
	return runComparison{
		Baseline:  base,
		Candidate: next,
		Delta: runComparisonDelta{
			DurationMS:               next.DurationMS - base.DurationMS,
			OutputChars:              next.OutputChars - base.OutputChars,
			TrainRows:                next.TrainRows - base.TrainRows,
			EvalRows:                 next.EvalRows - base.EvalRows,
			CachedTensors:            runCachedTensors(next.Runtime) - runCachedTensors(base.Runtime),
			CachedTensorBytes:        runCachedTensorBytes(next.Runtime) - runCachedTensorBytes(base.Runtime),
			EvalCoveragePercentage:   runCoveragePercentage(next.EvalCoverage) - runCoveragePercentage(base.EvalCoverage),
			UniqueCoveragePercentage: runUniqueCoveragePercentage(next.EvalCoverage) - runUniqueCoveragePercentage(base.EvalCoverage),
		},
	}
}

func runComparisonItemFromEntry(entry runs.Entry) runComparisonItem {
	target := entry.Model
	if target == "" {
		target = entry.Path
	}
	item := runComparisonItem{
		ID:          entry.ID,
		Command:     entry.Command,
		Backend:     entry.Backend,
		Target:      target,
		Status:      runEntryStatus(entry),
		DurationMS:  entry.DurationMS,
		OutputChars: len([]rune(entry.Output)),
	}
	if entry.Training != nil {
		item.TrainRows = entry.Training.TrainRows
		item.EvalRows = entry.Training.EvalRows
		item.EvalCoverage = runComparisonEvalCoverageFromEntry(entry.Training.EvalCoverage)
	}
	item.Runtime = runComparisonRuntimeFromEntry(entry.Runtime)
	return item
}

func runComparisonRuntimeFromEntry(stats *nego.RuntimeStats) *runComparisonRuntime {
	if stats == nil {
		return nil
	}
	return &runComparisonRuntime{
		Backend:             stats.Backend,
		Device:              stats.Device,
		TensorCacheEnabled:  stats.TensorCacheEnabled,
		CachedTensors:       stats.CachedTensors,
		CachedTensorBytes:   stats.CachedTensorBytes,
		MaxTensorCacheBytes: stats.MaxTensorCacheBytes,
		AdapterLoaded:       stats.AdapterLoaded,
	}
}

func runComparisonEvalCoverageFromEntry(coverage *runs.EvalCoverage) *runComparisonEvalCoverage {
	if coverage == nil {
		return nil
	}
	return &runComparisonEvalCoverage{
		Rows:                     coverage.Rows,
		Tokens:                   coverage.Tokens,
		CoveredTokens:            coverage.CoveredTokens,
		Coverage:                 coverage.Coverage,
		CoveragePercentage:       coverage.Coverage * 100,
		UniqueTokens:             coverage.UniqueTokens,
		CoveredUniqueTokens:      coverage.CoveredUniqueTokens,
		UniqueCoverage:           coverage.UniqueCoverage,
		UniqueCoveragePercentage: coverage.UniqueCoverage * 100,
	}
}

func runEntryStatus(entry runs.Entry) string {
	if entry.Error != "" {
		return "error"
	}
	return "ok"
}

func writeRunsComparison(w io.Writer, comparison runComparison) {
	fmt.Fprintln(w, "Run comparison")
	writeRunComparisonItem(w, "Baseline", comparison.Baseline)
	writeRunComparisonItem(w, "Candidate", comparison.Candidate)
	fmt.Fprintf(w, "Duration delta: %s\n", signedDurationMS(comparison.Delta.DurationMS))
	if comparison.Baseline.OutputChars > 0 || comparison.Candidate.OutputChars > 0 {
		fmt.Fprintf(w, "Output chars:   %d -> %d (%+d)\n", comparison.Baseline.OutputChars, comparison.Candidate.OutputChars, comparison.Delta.OutputChars)
	}
	if comparison.Baseline.TrainRows > 0 || comparison.Candidate.TrainRows > 0 {
		fmt.Fprintf(w, "Train rows:     %d -> %d (%+d)\n", comparison.Baseline.TrainRows, comparison.Candidate.TrainRows, comparison.Delta.TrainRows)
	}
	if comparison.Baseline.EvalRows > 0 || comparison.Candidate.EvalRows > 0 {
		fmt.Fprintf(w, "Eval rows:      %d -> %d (%+d)\n", comparison.Baseline.EvalRows, comparison.Candidate.EvalRows, comparison.Delta.EvalRows)
	}
	if comparison.Baseline.Runtime != nil || comparison.Candidate.Runtime != nil {
		fmt.Fprintln(w, "Runtime cache:")
		fmt.Fprintf(w, "  Backend:      %s -> %s\n", formatRunRuntimeBackend(comparison.Baseline.Runtime), formatRunRuntimeBackend(comparison.Candidate.Runtime))
		fmt.Fprintf(w, "  Device:       %s -> %s\n", formatRunRuntimeDevice(comparison.Baseline.Runtime), formatRunRuntimeDevice(comparison.Candidate.Runtime))
		fmt.Fprintf(w, "  Tensors:      %d -> %d (%+d)\n", runCachedTensors(comparison.Baseline.Runtime), runCachedTensors(comparison.Candidate.Runtime), comparison.Delta.CachedTensors)
		fmt.Fprintf(w, "  Cached bytes: %s -> %s (%s)\n", humanBytesUint(runCachedTensorBytesUint(comparison.Baseline.Runtime)), humanBytesUint(runCachedTensorBytesUint(comparison.Candidate.Runtime)), signedBytes(comparison.Delta.CachedTensorBytes))
	}
	if comparison.Baseline.EvalCoverage != nil || comparison.Candidate.EvalCoverage != nil {
		fmt.Fprintln(w, "Eval coverage:")
		fmt.Fprintf(w, "  Tokens:       %s -> %s (%+.2f pp)\n", formatRunCoverage(comparison.Baseline.EvalCoverage), formatRunCoverage(comparison.Candidate.EvalCoverage), comparison.Delta.EvalCoveragePercentage)
		fmt.Fprintf(w, "  Unique tokens: %s -> %s (%+.2f pp)\n", formatRunUniqueCoverage(comparison.Baseline.EvalCoverage), formatRunUniqueCoverage(comparison.Candidate.EvalCoverage), comparison.Delta.UniqueCoveragePercentage)
	}
}

func writeRunComparisonItem(w io.Writer, label string, item runComparisonItem) {
	fmt.Fprintf(w, "%s: %s %s %s %dms\n", label, item.ID, item.Command, item.Status, item.DurationMS)
	if item.Backend != "" || item.Target != "" {
		fmt.Fprintf(w, "  %s %s\n", item.Backend, item.Target)
	}
}

func signedDurationMS(value int64) string {
	if value >= 0 {
		return fmt.Sprintf("+%dms", value)
	}
	return fmt.Sprintf("%dms", value)
}

func runCachedTensors(stats *runComparisonRuntime) int {
	if stats == nil {
		return 0
	}
	return stats.CachedTensors
}

func runCachedTensorBytes(stats *runComparisonRuntime) int64 {
	if stats == nil {
		return 0
	}
	if stats.CachedTensorBytes > uint64(^uint64(0)>>1) {
		return int64(^uint64(0) >> 1)
	}
	return int64(stats.CachedTensorBytes)
}

func runCachedTensorBytesUint(stats *runComparisonRuntime) uint64 {
	if stats == nil {
		return 0
	}
	return stats.CachedTensorBytes
}

func formatRunRuntimeBackend(stats *runComparisonRuntime) string {
	if stats == nil || stats.Backend == "" {
		return "not available"
	}
	return stats.Backend
}

func formatRunRuntimeDevice(stats *runComparisonRuntime) string {
	if stats == nil || stats.Device == "" {
		return "not available"
	}
	return stats.Device
}

func signedBytes(value int64) string {
	if value >= 0 {
		return "+" + humanBytesUint(uint64(value))
	}
	return "-" + humanBytesUint(uint64(-value))
}

func runCoveragePercentage(coverage *runComparisonEvalCoverage) float64 {
	if coverage == nil {
		return 0
	}
	return coverage.CoveragePercentage
}

func runUniqueCoveragePercentage(coverage *runComparisonEvalCoverage) float64 {
	if coverage == nil {
		return 0
	}
	return coverage.UniqueCoveragePercentage
}

func formatRunCoverage(coverage *runComparisonEvalCoverage) string {
	if coverage == nil {
		return "not available"
	}
	return fmt.Sprintf("%d/%d %.2f%%", coverage.CoveredTokens, coverage.Tokens, coverage.CoveragePercentage)
}

func formatRunUniqueCoverage(coverage *runComparisonEvalCoverage) string {
	if coverage == nil {
		return "not available"
	}
	return fmt.Sprintf("%d/%d %.2f%%", coverage.CoveredUniqueTokens, coverage.UniqueTokens, coverage.UniqueCoveragePercentage)
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
	if entry.Runtime != nil {
		writeRuntimeRunInfo(w, *entry.Runtime)
	}
	if entry.Training != nil {
		writeTrainingRunInfo(w, *entry.Training)
	}
}

func writeRuntimeRunInfo(w io.Writer, info nego.RuntimeStats) {
	fmt.Fprintln(w, "Runtime:")
	if info.Backend != "" {
		fmt.Fprintf(w, "  Backend:      %s\n", info.Backend)
	}
	if info.Device != "" {
		fmt.Fprintf(w, "  Device:       %s\n", info.Device)
	}
	fmt.Fprintf(w, "  Cache:        %s\n", enabledDisabled(info.TensorCacheEnabled))
	fmt.Fprintf(w, "  Tensors:      %d\n", info.CachedTensors)
	fmt.Fprintf(w, "  Cached bytes: %s\n", humanBytesUint(info.CachedTensorBytes))
	if info.MaxTensorCacheBytes > 0 {
		fmt.Fprintf(w, "  Cache limit:  %s\n", humanBytesUint(info.MaxTensorCacheBytes))
	}
	fmt.Fprintf(w, "  Adapter:      %s\n", yesNo(info.AdapterLoaded))
	if info.ExperimentalGeneration {
		fmt.Fprintln(w, "  Generation:   experimental")
	}
}

func writeTrainingRunInfo(w io.Writer, info runs.Training) {
	fmt.Fprintln(w, "Training:")
	if info.Method != "" {
		fmt.Fprintf(w, "  Method:        %s\n", info.Method)
	}
	if info.Artifact != "" {
		fmt.Fprintf(w, "  Artifact:      %s\n", info.Artifact)
	}
	if info.DatasetFormat != "" {
		fmt.Fprintf(w, "  Dataset:       %s\n", info.DatasetFormat)
	}
	if info.TrainFile != "" {
		fmt.Fprintf(w, "  Train file:    %s\n", info.TrainFile)
	}
	if info.EvalFile != "" {
		fmt.Fprintf(w, "  Eval file:     %s\n", info.EvalFile)
	}
	if info.OutputDir != "" {
		fmt.Fprintf(w, "  Output dir:    %s\n", info.OutputDir)
	}
	if info.AdapterPath != "" {
		fmt.Fprintf(w, "  Adapter:       %s\n", info.AdapterPath)
	}
	if info.ManifestPath != "" {
		fmt.Fprintf(w, "  Manifest:      %s\n", info.ManifestPath)
	}
	if info.DryRun {
		fmt.Fprintln(w, "  Dry run:       yes")
	}
	if info.LearningRate > 0 {
		fmt.Fprintf(w, "  Learning rate: %.6g\n", info.LearningRate)
	}
	if info.Epochs > 0 {
		fmt.Fprintf(w, "  Epochs:        %d\n", info.Epochs)
	}
	if info.MaxContext > 0 {
		fmt.Fprintf(w, "  Max context:   %d\n", info.MaxContext)
	}
	if info.TrainRows > 0 {
		fmt.Fprintf(w, "  Train rows:    %d\n", info.TrainRows)
	}
	if info.EvalRows > 0 {
		fmt.Fprintf(w, "  Eval rows:     %d\n", info.EvalRows)
	}
	if info.DuplicateRows > 0 {
		fmt.Fprintf(w, "  Duplicates:    %d\n", info.DuplicateRows)
	}
	if info.VocabSize > 0 {
		fmt.Fprintf(w, "  Vocab size:    %d\n", info.VocabSize)
	}
	if info.EvalCoverage != nil {
		fmt.Fprintf(w, "  Eval coverage: %d/%d tokens (%.2f%%), %d/%d unique (%.2f%%)\n",
			info.EvalCoverage.CoveredTokens,
			info.EvalCoverage.Tokens,
			info.EvalCoverage.Coverage*100,
			info.EvalCoverage.CoveredUniqueTokens,
			info.EvalCoverage.UniqueTokens,
			info.EvalCoverage.UniqueCoverage*100,
		)
	}
	if len(info.TopTokenIDs) > 0 {
		fmt.Fprintf(w, "  Top token IDs: %s\n", joinInts(info.TopTokenIDs))
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
	var withStatus bool

	fs := flag.NewFlagSet("models list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cacheDir, "cache-dir", "", "cache directory")
	fs.BoolVar(&jsonOutput, "json", false, "write JSON result")
	fs.BoolVar(&withStatus, "status", false, "include local artifact format and run/train readiness")
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
	if withStatus {
		fmt.Fprintln(tw, "REPO\tREVISION\tFORMAT\tRUN\tTRAIN\tSIZE\tFILES\tLOCAL PATH")
	} else {
		fmt.Fprintln(tw, "REPO\tREVISION\tSIZE\tFILES\tLOCAL PATH")
	}
	for _, entry := range entries {
		localPath := modelEntryLocalPath(entry)
		if withStatus {
			status := modelEntryStatus(localPath)
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n", entry.RepoID, entry.Revision, status.format, status.run, status.train, humanBytes(entry.TotalSize), entry.FileCount, localPath)
		} else {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\n", entry.RepoID, entry.Revision, humanBytes(entry.TotalSize), entry.FileCount, localPath)
		}
	}
	_ = tw.Flush()
	return 0
}

type modelEntryStatusResult struct {
	format string
	run    string
	train  string
}

func modelEntryLocalPath(entry registry.Entry) string {
	if entry.LocalDir != "" {
		return entry.LocalDir
	}
	return entry.SnapshotPath
}

func modelEntryStatus(path string) modelEntryStatusResult {
	if strings.TrimSpace(path) == "" {
		return modelEntryStatusResult{format: "missing", run: "not ready", train: "not ready"}
	}
	artifact, err := modelinfo.Resolve(path)
	if err != nil {
		return modelEntryStatusResult{format: "missing", run: "not ready", train: "not ready"}
	}
	run := artifact.RecommendedRunBackend
	if run == "" {
		run = "not ready"
	}
	train := artifact.RecommendedTrainBackend
	if train == "" {
		train = "not ready"
	}
	format := string(artifact.Format)
	if format == "" {
		format = "unknown"
	}
	return modelEntryStatusResult{format: format, run: run, train: train}
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

func humanBytesUint(n uint64) string {
	const maxInt64 = uint64(1<<63 - 1)
	if n <= maxInt64 {
		return humanBytes(int64(n))
	}
	return fmt.Sprintf("%d B", n)
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
