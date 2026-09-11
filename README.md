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
- Safetensors header, tensor metadata, and F32/F16/BF16 tensor buffer loading
- Hugging Face Qwen/Llama safetensors weight manifest checks
- Model artifact resolution for local run/train compatibility checks
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

For current implementation status and remaining pure-Go runtime gaps, read [docs/status.md](docs/status.md).

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
artifact, err := modelinfo.Resolve("./models/qwen3")
```

```bash
nego inspect ./models/qwen3
nego inspect ./models/qwen3-gguf --json
nego check ./models/qwen3-gguf
nego memory ./models/qwen3 --context 4096
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

The tokenizer helper supports WordLevel vocab maps, added tokens, BPE merge rules, and Unigram/SentencePiece-style vocab arrays for development workflows.

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

For local or remote auto-selection, leave `Backend` empty:

```go
model, err := nego.LoadModel(ctx, nego.ModelOptions{
    Path: "./models/qwen3-gguf",
})
```

Nego resolves GGUF artifacts to the experimental pure-Go `native` backend when possible. Hugging Face safetensors downloads resolve to the experimental pure-Go `native-hf` backend and can be used for native token-bias adapter training. Full LoRA/backprop training and production-quality local generation are still being built.

```go
import _ "github.com/gakon/nego-ai/backends/llama"
```

```bash
nego backends list
nego backends info llama.cpp
nego backends info native
nego backends info native-hf
nego run ./models/qwen3-gguf "Hello"
nego chat ./models/qwen3-gguf "Hello"
nego serve ./models/qwen3 --addr :8080
nego run --backend llama.cpp ./models/qwen3-gguf "Hello"
nego run ./models/qwen3 "Hello"
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
- Reads GGUF tokenizer vocabulary and BPE merge metadata in Go.
- Encodes and decodes text with a GGUF vocabulary tokenizer path.
- Decodes GGUF token IDs into text for native generation plumbing.
- Reads raw GGUF tensor bytes in Go.
- Includes early CPU tensor math primitives for F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K data.
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
- Includes deterministic temperature, top-k, top-p, repeat penalty, EOS-aware sampling, and native stream plumbing.
- Loads token-bias adapters produced by native training.
- Does not run production GGUF inference for large Qwen/Llama models yet.

The native backend is the foundation for pure-Go chat/train/share. The next phases are real Qwen/Llama compatibility, performance work, and broader quantized kernels.

## Pure-Go Hugging Face Runtime

Nego also includes an experimental `native-hf` backend for Hugging Face safetensors directories:

```go
import _ "github.com/gakon/nego-ai/backends/nativehf"

model, err := nego.LoadModel(ctx, nego.ModelOptions{
    Backend: "native-hf",
    Path:    "./models/qwen3",
})
```

Current status: it loads safetensors metadata, tensor readers, selected F32/F16/BF16 tensors as float32, HF Qwen/Llama weight manifests, tokenizer.json, native adapters, early float32 math primitives, prompt decode state, KV cache history, repeat-penalty/top-p/temperature sampling, experimental generation, and streaming chat. Full production-quality autoregressive generation for Qwen/Llama models is still in progress.

## Local llama.cpp runtime

```bash
NEGO_LLAMA_CLI=/path/to/llama-cli nego run --backend llama.cpp ./models/qwen3-gguf "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego run --backend llama.cpp ./models/qwen3-gguf "Hello" --threads 8 --ctx-size 4096 --gpu-layers 32
NEGO_LLAMA_CLI=/path/to/llama-cli nego run --backend llama.cpp ./models/qwen3-gguf "Hello" --gpu full --flash-attn
NEGO_LLAMA_CLI=/path/to/llama-cli nego run --backend llama.cpp ./models/qwen3-gguf "Hello" --gpu full --main-gpu 0 --tensor-split 3,1 --split-mode layer
NEGO_LLAMA_CLI=/path/to/llama-cli nego run --backend llama.cpp ./models/qwen3-gguf "Hello" --log runs.jsonl
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat --backend llama.cpp ./models/qwen3-gguf "Hello"
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat --backend llama.cpp ./models/qwen3-gguf --interactive
NEGO_LLAMA_CLI=/path/to/llama-cli nego chat --backend llama.cpp ./models/qwen3-gguf --interactive --session chats/qwen.json
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

The local server supports OpenAI-compatible non-streaming and streaming responses on `/v1/completions` and `/v1/chat/completions`.

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
nego dataset tokens data.jsonl --model ./models/qwen3 --format chat --max-context 4096
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
nego train capabilities ./models/qwen3
nego train init --base-model ./models/qwen3 --train-file examples/5.train/train.jsonl --eval-file examples/5.train/test.jsonl --dataset-format completion --max-context 4096 --output-dir ./outputs/qwen3-lora --out examples/5.train/train-job.json
nego train check examples/5.train/train-job.json
nego train validate examples/5.train/train-job.json
nego train job.json
```

Native token-bias adapter training is available as an early pure-Go path for GGUF or Hugging Face safetensors downloads:

```bash
nego train capabilities ./models/qwen3
nego train native ./models/qwen3 --train-file examples/5.train/train.jsonl --dataset-format completion --max-context 4096 --out ./outputs/qwen3-token-bias
nego train native ./models/qwen3-gguf --train-file examples/5.train/train.jsonl --dataset-format completion --max-context 4096 --out ./outputs/qwen3-token-bias
nego run --native ./models/qwen3-gguf "Hello" --adapter ./outputs/qwen3-token-bias/adapter.json --max-tokens 16
```

```go
model, err := nego.LoadModel(ctx, nego.ModelOptions{
    Backend: "native",
    Path:    "./models/qwen3-gguf",
    Options: map[string]string{
        "adapter_path": "./outputs/qwen3-token-bias/adapter.json",
    },
})
```

This writes a small token-bias adapter from the dataset. Full LoRA/backprop training remains planned.

## Share Preparation

```bash
nego share manifest ./outputs/qwen3-lora --out ./outputs/qwen3-lora/share-manifest.json --repo username/qwen3-lora --base-model Qwen/Qwen3-0.6B
nego share package ./outputs/qwen3-lora --out ./outputs/qwen3-lora.tar.gz --repo username/qwen3-lora --base-model Qwen/Qwen3-0.6B
nego share upload username/qwen3-lora ./outputs/qwen3-lora README.md --token $HF_TOKEN
```

`nego share upload` currently supports regular inline Hub commit uploads. Large model files that require Hugging Face LFS/Xet upload are still planned.

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
