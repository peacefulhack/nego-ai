# Nego

A Go-native toolkit for working with AI models, from downloading and caching model files to building chat, inference, and training workflows.

## Build

```bash
go build -o nego ./cmd/nego
nego version
```

## Examples

Runnable Go examples live in `examples/`.

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

```bash
nego inspect ./models/model.gguf
nego inspect ./models/model.gguf --json
```

## Tokenizer

```bash
nego tokenize ./models/qwen3 "hello world"
nego tokens ./models/qwen3 "hello world"
```

```go
ids, err := tok.EncodeBatch([]string{"hello", "world"})
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
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/gguf-model-dir "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/model.gguf "Hello" --threads 8 --ctx-size 4096 --gpu-layers 32
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/model.gguf "Hello" --log runs.jsonl
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat ./models/model.gguf "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat ./models/model.gguf --interactive
NEGO_LLAMA_CLI=/path/to/llama-cli nego serve ./models/model.gguf --addr :8080
```

When a GGUF file lives beside `chat_template.jinja` or `tokenizer_config.json`, the llama.cpp backend uses that template for chat prompts.

Run logs include prompts and outputs, so keep `runs.jsonl` private when working with sensitive data.

```bash
nego runs list runs.jsonl
nego runs show runs.jsonl <id>
```

## Embeddings

```bash
nego embed "hello world" --endpoint http://localhost:8080 --model text-embedding-model
```

```go
resp, err := nego.Embed(ctx, model, nego.EmbeddingRequest{
    Input: []string{"hello world"},
})
```

## RAG helpers

```go
chunks := rag.ChunkText(document, rag.ChunkOptions{MaxRunes: 800, Overlap: 80})
index := rag.NewIndex()
```

## Dataset utilities

```go
rows, err := datasets.ReadJSONL(reader)
train, test := datasets.Split(rows, 0.2, 42)
```

```bash
nego dataset inspect data.jsonl
nego dataset validate data.jsonl --format chat
nego dataset sample data.jsonl --n 5
nego dataset split data.jsonl --train-out train.jsonl --test-out test.jsonl --test-size 0.1
```

## Cache

```bash
nego cache usage
nego cache gc --yes
```

## Conversion helpers

```bash
nego convert gguf ./models/qwen3 --out ./models/qwen3.gguf --converter /path/to/convert_hf_to_gguf.py
```

## Training orchestration

```bash
nego train job.json
```

## Eval

```bash
nego eval suite.json
```

## Config workflow

```bash
nego run -f nego.json
nego chat -f nego.json
```
