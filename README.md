# Nego

A Go-native toolkit for working with AI models, from downloading and caching model files to building chat, inference, and training workflows.

## Build

```bash
go build -o nego ./cmd/nego
nego version
```

## Requirements

Core Nego features use Go only:

- Hugging Face-style downloads and cache management
- Model registry, inspection, tokenizer helpers, prompt rendering
- Dataset utilities, eval helpers, run logs, and training job orchestration

Some AI runtime and conversion features need third-party tools:

- Local chat/run/serve with GGUF models requires `llama-cli` from [llama.cpp](https://github.com/ggml-org/llama.cpp). Set `NEGO_LLAMA_CLI` when `llama-cli` is not on PATH.
- Converting Hugging Face `*.safetensors` models to GGUF requires the upstream `convert_hf_to_gguf.py` script from llama.cpp plus its Python dependencies. Set `NEGO_LLAMA_CONVERTER` or pass `--converter`.
- Quantizing GGUF files requires `llama-quantize` from llama.cpp.
- Fine-tuning is currently an external job runner. Nego can create and validate job JSON, but the actual trainer can be Python, shell, or another training stack you choose.

Nego does not vendor llama.cpp or its Python converter. This keeps the Go library lightweight and lets users choose a llama.cpp build that matches their CPU/GPU setup.

## Examples

Runnable Go examples live in `examples/`.

For an end-to-end numbered workflow, read [docs/workflow.md](docs/workflow.md). It walks through download, inspect, dataset preparation, context checks, chat, eval, training job JSON, conversion, serving, and planned sharing steps.

Numbered runnable examples live in [examples](examples):

```bash
go run ./examples/1.download
go run ./examples/3.prepare-dataset
go run ./examples/4.eval
go run ./examples/5.train
go run ./examples/8.share-trained-model
```

## Download models

```bash
nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3
nego download Qwen/Qwen3-0.6B tokenizer.json --local-dir ./models/qwen3
nego download Qwen/Qwen3-0.6B --include "*.safetensors" --exclude "*.msgpack"
nego download Qwen/Qwen3-0.6B --revision main --token "$HF_TOKEN"
nego download Qwen/Qwen3-0.6B --gguf --local-dir ./models/qwen3-gguf
nego download Qwen/Qwen3-0.6B --gguf --gguf-repo unsloth/Qwen3-0.6B-GGUF --quant Q4_K_M --local-dir ./models/qwen3-gguf
```

Regular Hugging Face downloads are useful for inspection, tokenization, and training jobs. Local chat through llama.cpp needs GGUF, so Nego warns when a downloaded model only contains safetensors.

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

gguf, err := hub.ResolveGGUFFile(ctx, hub.ResolveGGUFFileOptions{
    RepoID: "Qwen/Qwen3-0.6B",
    Quant:  "Q4_K_M",
})
path, err := hub.DownloadFile(ctx, hub.DownloadFileOptions{
    RepoID:   gguf.RepoID,
    Filename: gguf.Filename,
    LocalDir: "./models/qwen3-gguf",
})
```

```go
info, err := modelinfo.Inspect("./models/qwen3")
```

```bash
nego inspect ./models/qwen3
nego inspect ./models/qwen3-gguf --json
nego check ./models/qwen3-gguf
```

## Tokenizer

```bash
nego tokenize ./models/qwen3 "hello world"
nego tokens ./models/qwen3 "hello world"
nego context ./models/qwen3 "hello world" --max-context 4096
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
    Backend: "llama.cpp",
    Path:    "./models/qwen3-gguf",
    Options: map[string]string{"gpu": "auto"},
})
```

```go
import _ "github.com/gakon/nego-ai/backends/llama"
```

```bash
nego backends list
nego backends info llama.cpp
nego backends info native
nego run --native ./models/qwen3-gguf "Hello"
nego chat --native ./models/qwen3-gguf "Hello"
```

## Pure-Go native runtime

Nego includes an experimental `native` backend for the pure-Go runtime track:

```go
import _ "github.com/gakon/nego-ai/backends/native"

model, err := nego.LoadModel(ctx, nego.ModelOptions{
    Backend: "native",
    Path:    "./models/qwen3-gguf",
})
```

Current status:

- Loads GGUF files without `llama-cli`.
- Parses GGUF metadata and tensor directories in Go.
- Builds native model specs and standard tensor-name maps from GGUF metadata.
- Reports missing or mismatched native runtime tensors before forward-pass work.
- Loads per-block native runtime weights from GGUF tensor storage.
- Reads GGUF tokenizer vocabulary metadata in Go.
- Encodes and decodes text with a basic GGUF vocabulary tokenizer path.
- Decodes GGUF token IDs into text for native generation plumbing.
- Reads raw GGUF tensor bytes in Go.
- Includes early CPU tensor math primitives for F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and Q8_1 data.
- Includes RMSNorm, SiLU, and softmax primitives for transformer blocks.
- Includes residual/vector helpers and RoPE primitives for attention plumbing.
- Includes scaled dot-product attention and KV cache primitives.
- Includes single-step multi-head attention assembly for Q/K/V/O projections.
- Includes decode state plumbing that appends K/V vectors while processing prompt and generated tokens.
- Includes embedding lookup and output-logits helpers for generation plumbing.
- Includes generation option normalization and token sample/decode scaffolding.
- Includes prompt planning for native generation before the transformer forward pass lands.
- Includes linear projection and MLP helpers for transformer feed-forward blocks.
- Includes a float32 transformer block scaffold for attention, residuals, and MLP.
- Runs a single-token float32 forward path when a small GGUF has supported tensors.
- Runs an early multi-token generate/chat loop over the float32 forward path for supported tiny GGUF fixtures.
- Loads supported GGUF tensors into float32 buffers for native runtime prototyping.
- Caches loaded float32 tensor buffers per model instance for native forward experiments.
- Includes deterministic temperature, top-k, and top-p sampling primitives.
- Does not run production GGUF inference for large models or K-quant formats such as Q4_K_M yet.

The native backend is the foundation for pure-Go chat/train/share. The next phases are real Qwen/Llama compatibility, performance work, and broader quantized kernels.

## Local llama.cpp runtime

```bash
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/qwen3-gguf "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/qwen3-gguf "Hello" --threads 8 --ctx-size 4096 --gpu-layers 32
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/qwen3-gguf "Hello" --gpu full --flash-attn
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/qwen3-gguf "Hello" --gpu full --main-gpu 0 --tensor-split 3,1 --split-mode layer
NEGO_LLAMA_CLI=/path/to/llama-cli nego run ./models/qwen3-gguf "Hello" --log runs.jsonl
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat ./models/qwen3-gguf "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat ./models/qwen3-gguf --interactive
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat ./models/qwen3-gguf --interactive --session chats/qwen.json
NEGO_LLAMA_CLI=/path/to/llama-cli nego serve ./models/qwen3-gguf --addr :8080
```

When a GGUF file lives beside `chat_template.jinja` or `tokenizer_config.json`, the llama.cpp backend uses that template for chat prompts.

Run logs include prompts and outputs, so keep `runs.jsonl` private when working with sensitive data.
Chat sessions also include message history, so keep session JSON files private.

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
nego dataset convert data.csv --out data.jsonl --select prompt,completion --require prompt,completion
nego dataset filter data.jsonl --where split=train --out train-only.jsonl
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
nego convert gguf ./models/qwen3 --out ./models/qwen3.gguf --converter /path/to/convert_hf_to_gguf.py --python python
NEGO_LLAMA_CONVERTER=/path/to/convert_hf_to_gguf.py nego convert gguf ./models/qwen3 --out ./models/qwen3.gguf --python python
```

Nego does not natively convert Hugging Face safetensors to GGUF yet. Use `nego download <repo> --gguf` when a GGUF repo exists, or use `nego convert gguf` with a llama.cpp converter.

## Training orchestration

Use the Hugging Face-style model directory (`./models/qwen3`) for fine-tuning jobs. Use the GGUF directory (`./models/qwen3-gguf`) for local chat/runtime.

```bash
nego train init --base-model ./models/qwen3 --train-file examples/5.train/train.jsonl --eval-file examples/5.train/test.jsonl --output-dir ./outputs/qwen3-lora --out examples/5.train/train-job.json
nego train validate examples/5.train/train-job.json
nego train job.json
```

## Share Preparation

```bash
nego share manifest ./outputs/qwen3-lora --out ./outputs/qwen3-lora/share-manifest.json --repo username/qwen3-lora --base-model Qwen/Qwen3-0.6B
```

## Eval

```bash
nego eval suite.json
nego eval suite.json --json > results.json
nego eval report results.json
nego eval compare baseline.json candidate.json
```

## Config workflow

```bash
nego run -f nego.json
nego chat -f nego.json
```
