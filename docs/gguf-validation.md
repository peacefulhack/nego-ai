# Pure-Go Qwen3 GGUF Walkthrough

This path was smoke-tested with `unsloth/Qwen3-0.6B-GGUF`, file
`Qwen3-0.6B-Q4_K_M.gguf` (396,705,472 bytes). File SHA-256:
`ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`.

All commands run from the repository root. Inference and token-bias training use
Go on CPU, with no Python, `llama-cli`, or remote inference service. Downloading
requires internet access; subsequent commands operate locally.

## 1. Download

```powershell
go run ./cmd/nego download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf --local-dir ./models/qwen3-gguf
```

The upstream repository can change. Compare the local file hash with the value
above when reproducing these results. Model files are not committed to Nego.

## 2. Check and Chat

```powershell
go run ./cmd/nego status ./models/qwen3-gguf --runtime-smoke --runtime-stats
go run ./cmd/nego run ./models/qwen3-gguf 'The capital of France is' --native --max-tokens 16 --temperature 0
go run ./cmd/nego chat ./models/qwen3-gguf 'What is Go? /no_think' --native --max-tokens 96 --temperature 0
```

The completion began with ` Paris.` and chat produced a readable explanation of
Go. These are smoke tests, not factual-quality benchmarks. A token limit can
truncate the answer. The decoded tensor cache used approximately 2.2 GiB, much
larger than the quantized file; allow extra RAM for temporary buffers and context.

## 3. Prepare Data

Use `examples/5.train/train.jsonl` and `examples/5.train/test.jsonl`. The completion
format contains one JSON object per line, with `prompt` and `completion` strings:

```jsonl
{"prompt":"What is Go?","completion":"Go is a programming language."}
{"prompt":"What is a goroutine?","completion":"A goroutine is a lightweight concurrent function execution."}
```

Token-bias training counts target tokens and saves sampling biases. It does not
teach new reasoning skills, change transformer weights, or perform LoRA/backprop.

## 4. Train the Adapter

Choose a new output directory for each experiment:

```powershell
go run ./cmd/nego train native ./models/qwen3-gguf --train-file examples/5.train/train.jsonl --eval-file examples/5.train/test.jsonl --dataset-format completion --out ./outputs/qwen3-gguf-token-bias
```

The local sample run processed two training rows, counted 28 target tokens, and
updated 26 token biases. Eval token coverage was 4/13, not an accuracy score.
Output includes `adapter.json`, `manifest.json`, and a reuse guide.

## 5. Chat With That Adapter

```powershell
go run ./cmd/nego chat ./outputs/qwen3-gguf-token-bias 'What is Go? /no_think' --native --max-tokens 48 --temperature 0
```

The manifest resolves the same base GGUF model. Keep that base file available;
the adapter is not a standalone model and must use its original vocabulary.

## 6. Package for Sharing

```powershell
go run ./cmd/nego share package ./outputs/qwen3-gguf-token-bias --out ./outputs/qwen3-gguf-token-bias.zip
```

Packaging was verified locally. It does not publish to a Hub or bundle the base
weights. Recipients need the matching base model and a resolvable base-model path.

## 7. Run the Opt-In Tokenizer Check

```powershell
$env:NEGO_QWEN_GGUF = (Resolve-Path ./models/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf).Path
go test ./modelinfo -run QwenGGUFLocalReference -count=1
```

All 24 cases matched independent Hugging Face token-ID fixtures with NFC disabled
to represent the GGUF Qwen profile. They cover multilingual text, whitespace,
Unicode bytes, and chat markers. Normal tests use tiny local fixtures only.

## Remaining Limits

- No GPU execution or native LoRA/backprop yet.
- Other architectures, quantization combinations, and scaled/partial RoPE need validation.
- This is not numerical parity certification against llama.cpp.
- GGUF Qwen does not apply the HF tokenizer's NFC normalization implicitly.
- Context-length and tensor-cache estimates are not a guarantee that a model fits in RAM.

## Attention Allocation Benchmark

```powershell
go test ./backends/native -run '^$' -bench BenchmarkCachedGroupedAttention -benchmem
```

The synthetic fixture uses eight query heads, two KV heads, head dimension 16,
and a fixed history length. On the development Windows/amd64 machine, reusing
per-group history views changed the 1,024-token case from approximately 517,842
to 190,672 allocated bytes per call (63 to 51 allocations). This measures temporary
attention allocations, not total model RAM or end-to-end tokens per second.

Implementation references: [llama.cpp RoPE layouts](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)
and [Qwen pre-tokenization](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp).
