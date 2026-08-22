# Nego

A Go-native toolkit for working with AI models, from downloading and caching model files to building chat, inference, and training workflows.

## Download models

```bash
nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3
nego download Qwen/Qwen3-0.6B tokenizer.json --local-dir ./models/qwen3
nego download Qwen/Qwen3-0.6B --include "*.safetensors" --exclude "*.msgpack"
nego download Qwen/Qwen3-0.6B --revision main --token "$HF_TOKEN"
```

## Go API

```go
path, err := hub.DownloadFile(ctx, hub.DownloadFileOptions{
    RepoID:   "Qwen/Qwen3-0.6B",
    Filename: "tokenizer.json",
    LocalDir: "./models/qwen3",
})
```

```go
path, err := hub.DownloadSnapshot(ctx, hub.DownloadSnapshotOptions{
    RepoID:   "Qwen/Qwen3-0.6B",
    LocalDir: "./models/qwen3",
    Include: []string{"*.safetensors", "config.json", "tokenizer.json"},
})
```

```go
info, err := modelinfo.Inspect("./models/qwen3")
```

## Tokenizer

```bash
nego tokenize ./models/qwen3 "hello world"
nego tokens ./models/qwen3 "hello world"
```

## Prompt rendering

```bash
nego prompt ./models/qwen3 --system "You are helpful" --user "Hello"
```

## Runtime API

```go
model, err := nego.LoadModel(ctx, nego.ModelOptions{
    Backend: "openai-compatible",
    Model:   "qwen3",
})
```

```go
import _ "github.com/gakon/nego-ai/backends/openai"
```

## Local llama.cpp runtime

```bash
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/model.gguf "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat ./models/model.gguf "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego serve ./models/model.gguf --addr :8080
```
