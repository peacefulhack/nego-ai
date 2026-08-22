package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	nego "github.com/gakon/nego-ai"
	_ "github.com/gakon/nego-ai/backends/llama"
	_ "github.com/gakon/nego-ai/backends/openai"
	"github.com/gakon/nego-ai/chattemplate"
	"github.com/gakon/nego-ai/evals"
	"github.com/gakon/nego-ai/hub"
	"github.com/gakon/nego-ai/internal/cache"
	"github.com/gakon/nego-ai/internal/registry"
	"github.com/gakon/nego-ai/server"
	"github.com/gakon/nego-ai/tokenizer"
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
	case "tokenize":
		return runTokenize(args[1:], stdout, stderr)
	case "tokens":
		return runTokens(args[1:], stdout, stderr)
	case "prompt":
		return runPrompt(args[1:], stdout, stderr)
	case "run":
		return runModel(args[1:], stdout, stderr)
	case "chat":
		return runChat(args[1:], stdout, stderr)
	case "serve":
		return runServe(args[1:], stdout, stderr)
	case "embed":
		return runEmbed(args[1:], stdout, stderr)
	case "eval":
		return runEval(args[1:], stdout, stderr)
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
	case "force", "local-files-only", "quiet", "json", "yes", "no-generation-prompt":
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
	fmt.Fprintln(w, "  nego models remove <repo-id> --yes [flags]")
	fmt.Fprintln(w, "  nego tokenize <model-path> <text> [flags]")
	fmt.Fprintln(w, "  nego tokens <model-path> <text>")
	fmt.Fprintln(w, "  nego prompt <model-path> --user <text> [flags]")
	fmt.Fprintln(w, "  nego run <model-path> <prompt> [flags]")
	fmt.Fprintln(w, "  nego chat <model-path> <message> [flags]")
	fmt.Fprintln(w, "  nego serve <model-path> [flags]")
	fmt.Fprintln(w, "  nego embed <text> --endpoint <url> --model <name> [flags]")
	fmt.Fprintln(w, "  nego eval <suite.json> [flags]")
}

type evalConfig struct {
	Backend  string       `json:"backend"`
	Path     string       `json:"path"`
	Endpoint string       `json:"endpoint"`
	Model    string       `json:"model"`
	APIKey   string       `json:"api_key"`
	Cases    []evals.Case `json:"cases"`
}

func runEval(args []string, stdout, stderr io.Writer) int {
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
	prompt, err := chattemplate.Render(positionals[0], messages, chattemplate.Options{AddGenerationPrompt: !noGenerationPrompt})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, prompt)
	return 0
}

func runModel(args []string, stdout, stderr io.Writer) int {
	var backend string
	var maxTokens int
	var configFile string
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "llama.cpp", "runtime backend")
	fs.IntVar(&maxTokens, "max-tokens", 0, "maximum tokens to generate")
	fs.StringVar(&configFile, "f", "", "run config file")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
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
	if cfg.MaxTokens > 0 && maxTokens == 0 {
		maxTokens = cfg.MaxTokens
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
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{Backend: backend, Path: path, Endpoint: cfg.Endpoint, Model: cfg.Model, APIKey: cfg.APIKey})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	defer model.Close()
	out, err := model.Generate(context.Background(), nego.GenerateRequest{Prompt: prompt, MaxTokens: maxTokens})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, out.Text)
	return 0
}

func runChat(args []string, stdout, stderr io.Writer) int {
	var backend string
	var system string
	var maxTokens int
	var configFile string
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "llama.cpp", "runtime backend")
	fs.StringVar(&system, "system", "", "system message")
	fs.IntVar(&maxTokens, "max-tokens", 0, "maximum tokens to generate")
	fs.StringVar(&configFile, "f", "", "chat config file")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
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
	if cfg.System != "" && system == "" {
		system = cfg.System
	}
	if cfg.MaxTokens > 0 && maxTokens == 0 {
		maxTokens = cfg.MaxTokens
	}
	path := cfg.Path
	if len(positionals) > 0 {
		path = positionals[0]
	}
	messages := append([]nego.Message(nil), cfg.Messages...)
	if system != "" && len(messages) == 0 {
		messages = append(messages, nego.Message{Role: nego.RoleSystem, Content: system})
	}
	if len(positionals) > 1 {
		messages = append(messages, nego.Message{Role: nego.RoleUser, Content: strings.Join(positionals[1:], " ")})
	} else if cfg.Prompt != "" && len(messages) == 0 {
		messages = append(messages, nego.Message{Role: nego.RoleUser, Content: cfg.Prompt})
	}
	if path == "" || len(messages) == 0 {
		fmt.Fprintln(stderr, "usage: nego chat <model-path> <message> [flags]")
		return 2
	}
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{Backend: backend, Path: path, Endpoint: cfg.Endpoint, Model: cfg.Model, APIKey: cfg.APIKey})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	defer model.Close()
	resp, err := model.Chat(context.Background(), nego.ChatRequest{Messages: messages, MaxTokens: maxTokens})
	if err != nil {
		fmt.Fprintf(stderr, "nego: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, resp.Message.Content)
	return 0
}

type runtimeConfig struct {
	Backend   string         `json:"backend"`
	Path      string         `json:"path"`
	Endpoint  string         `json:"endpoint"`
	Model     string         `json:"model"`
	APIKey    string         `json:"api_key"`
	System    string         `json:"system"`
	Prompt    string         `json:"prompt"`
	Messages  []nego.Message `json:"messages"`
	MaxTokens int            `json:"max_tokens"`
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

func runServe(args []string, stdout, stderr io.Writer) int {
	var backend string
	var addr string
	var modelID string

	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&backend, "backend", "llama.cpp", "runtime backend")
	fs.StringVar(&addr, "addr", ":8080", "listen address")
	fs.StringVar(&modelID, "model", "nego-model", "served model id")
	parseArgs, positionals := splitFlags(args)
	if err := fs.Parse(parseArgs); err != nil {
		return 2
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "usage: nego serve <model-path> [flags]")
		return 2
	}
	model, err := nego.LoadModel(context.Background(), nego.ModelOptions{Backend: backend, Path: positionals[0]})
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
