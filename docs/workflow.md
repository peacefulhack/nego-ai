# Nego Workflow Guide

This guide shows the intended day-to-day Nego workflow from downloading a model to preparing data, chatting, evaluating, training, and preparing a model for sharing.

Some late workflow steps are marked as planned because Nego does not implement direct model upload or full native fine-tuning yet. The current training command runs an external training process from a JSON job file.

## Numbered Flow

1. Download a base model.
2. Inspect the downloaded model.
3. Prepare a dataset.
4. Check tokenizer and context budget.
5. Render a chat prompt.
6. Chat or run the model.
7. Run a baseline eval.
8. Train or fine-tune with a job JSON file.
9. Inspect and evaluate the trained output.
10. Convert or optimize the model.
11. Serve locally.
12. Package or share the model.

Runnable example code and supporting files live in numbered folders under `examples/`.

## 1. Download a Base Model

Download a full model snapshot into a local directory:

```bash
nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3
```

That downloads the original Hugging Face files, usually `*.safetensors`. Use this for inspection, tokenizer work, dataset checks, or training flows. Nego will warn that safetensors are not directly runnable by the local llama.cpp backend.

Download a llama.cpp-ready GGUF file when you want to run local chat immediately:

```bash
nego download Qwen/Qwen3-0.6B --gguf --local-dir ./models/qwen3-gguf
```

If the model does not have a same-owner `-GGUF` repo, choose a trusted GGUF repo explicitly:

```bash
nego download Qwen/Qwen3-0.6B \
  --gguf \
  --gguf-repo unsloth/Qwen3-0.6B-GGUF \
  --quant Q4_K_M \
  --local-dir ./models/qwen3-gguf
```

Download only one file:

```bash
nego download Qwen/Qwen3-0.6B tokenizer.json --local-dir ./models/qwen3
```

Download only selected files:

```bash
nego download Qwen/Qwen3-0.6B --include "*.safetensors" --exclude "*.msgpack" --local-dir ./models/qwen3
```

Download a private or gated model:

```bash
nego download meta-llama/Llama-3.2-1B --token "$HF_TOKEN" --local-dir ./models/llama
```

## 2. Inspect the Downloaded Model

Check model files, config, GGUF metadata, and model card metadata:

```bash
nego inspect ./models/qwen3
```

Get machine-readable metadata:

```bash
nego inspect ./models/qwen3 --json
```

Check runtime compatibility:

```bash
nego check ./models/qwen3
```

List available runtime backends:

```bash
nego backends list
nego backends info llama.cpp
nego backends info native
nego backends info openai-compatible
```

`native` is the pure-Go runtime track. It can load GGUF metadata, build model specs and tensor-name maps, report missing runtime tensors, read tokenizer vocabulary metadata, encode/decode text through a basic GGUF vocab path, plan prompt tokens for generation, read tensor directories and raw tensor bytes, load selected tensors as float32 buffers, run early CPU tensor math, activation, vector, RoPE, attention, KV cache, single-step multi-head attention, embedding, MLP, logits, and generation sampling primitives, and sample deterministically today, but full transformer inference is still under development.

## 3. Prepare a Dataset

Start with a CSV file:

```csv
prompt,completion,split
"What is Go?","Go is a programming language.",train
"Explain AI briefly.","AI is software that performs tasks requiring intelligence.",train
"Say hello.","Hello!",test
```

Inspect the dataset:

```bash
nego dataset inspect examples/3.prepare-dataset/completion-data.csv
```

Convert CSV to JSONL while keeping only the fields needed for completion training:

```bash
nego dataset convert examples/3.prepare-dataset/completion-data.csv \
  --out data/completion.jsonl \
  --select prompt,completion \
  --require prompt,completion
```

The output JSONL shape is:

```jsonl
{"prompt":"What is Go?","completion":"Go is a programming language."}
{"prompt":"Explain AI briefly.","completion":"AI is software that performs tasks requiring intelligence."}
```

Validate the JSONL dataset:

```bash
nego dataset validate data/completion.jsonl --format completion
```

Filter rows for a subset:

```bash
nego dataset filter examples/3.prepare-dataset/completion-data.csv \
  --where split=train \
  --select prompt,completion \
  --out data/train-only.jsonl
```

Split train/test:

```bash
nego dataset split data/completion.jsonl \
  --train-out data/train.jsonl \
  --test-out data/test.jsonl \
  --test-size 0.1
```

Sample a few rows:

```bash
nego dataset sample data/train.jsonl --n 5
```

## 4. Check Tokenizer and Context Budget

Count tokens:

```bash
nego tokens ./models/qwen3 "Explain Go in simple terms."
```

Show token IDs:

```bash
nego tokenize ./models/qwen3 "Explain Go in simple terms."
```

Check whether a prompt fits a context window:

```bash
nego context ./models/qwen3 "Explain Go in simple terms." --max-context 4096
```

Check a chat-shaped prompt:

```bash
nego context ./models/qwen3 \
  --system "You are helpful." \
  --user "Explain Go in simple terms." \
  --max-context 4096
```

## 5. Render a Chat Prompt

Render messages through the local model template:

```bash
nego prompt ./models/qwen3 \
  --system "You are helpful." \
  --user "Explain Go in simple terms."
```

This is useful before running a backend because you can inspect the exact prompt that will be sent to the runtime.

## 6. Chat or Run the Model

For local GGUF models, set `NEGO_LLAMA_CLI` if `llama-cli` is not on your PATH.

PowerShell:

```powershell
$env:NEGO_LLAMA_CLI = "C:\tools\llama.cpp\llama-cli.exe"
```

Unix shells:

```bash
export NEGO_LLAMA_CLI=/path/to/llama-cli
```

Run one prompt:

```bash
nego run ./models/qwen3-gguf "Explain Go in one paragraph."
```

Run with GPU acceleration when your `llama-cli` build supports CUDA, Metal, Vulkan, ROCm, or another llama.cpp GPU backend:

```bash
nego run ./models/qwen3-gguf "Explain Go in one paragraph." --gpu full --flash-attn
```

Force CPU-only:

```bash
nego run ./models/qwen3-gguf "Explain Go in one paragraph." --gpu off
```

Use explicit multi-GPU placement:

```bash
nego run ./models/qwen3-gguf "Explain Go in one paragraph." \
  --gpu full \
  --main-gpu 0 \
  --tensor-split 3,1 \
  --split-mode layer
```

Run one chat request:

```bash
nego chat ./models/qwen3-gguf "Hello"
```

Start interactive chat:

```bash
nego chat ./models/qwen3-gguf --interactive
```

Start interactive chat with a resumable session:

```bash
nego chat ./models/qwen3-gguf --interactive --session chats/qwen.json
```

Save a one-shot chat session:

```bash
nego chat ./models/qwen3-gguf "Hello" --save chats/hello.json
```

Session JSON shape:

```json
{
  "backend": "llama.cpp",
  "path": "./models/qwen3-gguf",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi!"
    }
  ]
}
```

Keep session files private because they include prompt and response history.

## 7. Run a Baseline Eval

Create an eval suite:

```json
{
  "backend": "llama.cpp",
  "path": "./models/qwen3-gguf",
  "cases": [
    {
      "name": "greeting",
      "prompt": "Say hello in one short sentence.",
      "contains": "hello",
      "max_tokens": 32
    },
    {
      "name": "json validity",
      "prompt": "Return JSON with ok=true.",
      "json_valid": true,
      "max_tokens": 64
    },
    {
      "name": "chat answer",
      "messages": [
        {
          "role": "system",
          "content": "You answer briefly."
        },
        {
          "role": "user",
          "content": "What language is used by Nego?"
        }
      ],
      "contains": "Go",
      "max_tokens": 64
    }
  ]
}
```

Run the suite and save a baseline report:

```bash
nego eval examples/4.eval/eval-suite.json --json > reports/baseline.json
```

Read the report:

```bash
nego eval report reports/baseline.json
```

## 8. Train or Fine-Tune

Current Nego training is an external process runner. It is useful for wrapping Python, shell scripts, or another training tool while keeping a consistent Nego workflow. Training uses the Hugging Face-style model from step 1 (`./models/qwen3`), not the GGUF runtime copy used for chat.

Create a job JSON from the downloaded model and prepared data:

```bash
nego train init \
  --base-model ./models/qwen3 \
  --train-file examples/5.train/train.jsonl \
  --eval-file examples/5.train/test.jsonl \
  --output-dir ./outputs/qwen3-lora \
  --out examples/5.train/train-job.json
```

Minimal `job.json` shape:

```json
{
  "name": "qwen3-lora",
  "method": "lora",
  "base_model": "./models/qwen3",
  "train_file": "examples/5.train/train.jsonl",
  "eval_file": "examples/5.train/test.jsonl",
  "output_dir": "./outputs/qwen3-lora",
  "command": "python",
  "args": [
    "scripts/train_lora.py",
    "--model",
    "./models/qwen3",
    "--train-file",
    "examples/5.train/train.jsonl",
    "--output-dir",
    "./outputs/qwen3-lora",
    "--eval-file",
    "examples/5.train/test.jsonl"
  ],
  "env": {
    "TOKENIZERS_PARALLELISM": "false"
  },
  "work_dir": "."
}
```

Run it:

```bash
nego train examples/5.train/train-job.json
```

The command prints stdout/stderr from the training process and exits non-zero if the process fails.

## 9. Inspect and Evaluate the Trained Output

Inspect the produced model directory:

```bash
nego inspect ./outputs/qwen3-sft
nego check ./outputs/qwen3-sft
```

Run the same eval suite against the trained model and compare with the baseline. Update the suite `path` to point at the trained model first, for example `./outputs/qwen3-sft.gguf`.

```bash
nego eval examples/4.eval/eval-suite.json --json > reports/trained.json
nego eval compare reports/baseline.json reports/trained.json
```

## 10. Convert or Optimize

Convert a Hugging Face-style model directory to GGUF using a llama.cpp converter. Nego does not natively transform safetensors into GGUF yet; it wraps the converter as a safe, explicit external process.

```bash
nego convert gguf ./outputs/qwen3-sft \
  --out ./outputs/qwen3-sft.gguf \
  --converter /path/to/convert_hf_to_gguf.py \
  --python python
```

You can also set the converter once:

```bash
export NEGO_LLAMA_CONVERTER=/path/to/convert_hf_to_gguf.py
nego convert gguf ./outputs/qwen3-sft --out ./outputs/qwen3-sft.gguf --python python
```

Optionally request an output type supported by your converter:

```bash
nego convert gguf ./outputs/qwen3-sft \
  --out ./outputs/qwen3-sft-q4.gguf \
  --converter /path/to/convert_hf_to_gguf.py \
  --python python \
  --quantize q4_0
```

## 11. Serve Locally

Serve a local model through Nego's HTTP server:

```bash
nego serve ./outputs/qwen3-sft.gguf --addr :8080 --gpu full --flash-attn
```

Then point an OpenAI-compatible client at:

```text
http://localhost:8080/v1/chat/completions
```

## 12. Package or Share the Model

Direct model packaging and hub upload commands are planned. For now, prepare the output directory manually:

```text
outputs/qwen3-sft/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors or adapter files
  README.md
```

Recommended model card front matter:

```yaml
---
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
base_model: Qwen/Qwen3-0.6B
datasets:
  - custom-dataset
tags:
  - fine-tuned
  - lora
  - nego
---
# qwen3-sft
```

Planned Nego commands:

```bash
nego model package ./outputs/qwen3-sft --out qwen3-sft.tar
nego hub upload ./outputs/qwen3-sft --repo username/qwen3-sft
```

## Current Support Summary

1. Download a base model: implemented.
2. Inspect the downloaded model: implemented.
3. Prepare a dataset: implemented.
4. Check tokenizer and context budget: implemented.
5. Render a chat prompt: implemented.
6. Chat or run the model: implemented.
7. Run a baseline eval: implemented.
8. Train or fine-tune with a job JSON file: implemented as an external job runner.
9. Inspect and evaluate the trained output: implemented.
10. Convert or optimize the model: implemented through an external GGUF converter.
11. Serve locally: implemented.
12. Package or share the model: planned.
