# Step 5: Train

This step uses the model downloaded in step 1 and the dataset files in this
folder.

1. Check the downloaded model can be used for training:

```bash
go run ./cmd/nego train capabilities ./models/qwen3
```

2. Use completion JSONL for a small supervised dataset:

```jsonl
{"prompt":"Say hello in Indonesian.","completion":"Halo!"}
{"prompt":"Name the language Nego is written in.","completion":"Go."}
```

Each row needs `prompt` and `completion`. Chat datasets should use a
`messages` array with `system`, `user`, and `assistant` roles.

3. Validate and preview the exact text used for training:

```bash
go run ./cmd/nego dataset validate examples/5.train/train.jsonl --format completion
go run ./cmd/nego dataset render examples/5.train/train.jsonl --model ./models/qwen3 --format completion --n 3
go run ./cmd/nego dataset tokens examples/5.train/train.jsonl --model ./models/qwen3 --format completion --max-context 4096
```

4. Generate and validate tokenizer fixture cases before training:

```bash
go run ./cmd/nego tokenize fixtures qwen \
  --model ./models/qwen3 \
  --out examples/5.train/tokenizer-fixtures.json

go run ./cmd/nego tokenize check ./models/qwen3 examples/5.train/tokenizer-fixtures.json
```

Use `basic`, `qwen`, or `llama` as the profile. Passing `--model` records
the local model's token IDs so the fixture becomes a stricter regression
baseline.

```bash
go run ./cmd/nego train native ./models/qwen3 \
  --train-file examples/5.train/train.jsonl \
  --dataset-format completion \
  --tokenize-check examples/5.train/tokenizer-fixtures.json \
  --dry-run
```

The fixture file can check token IDs with `tokens` or `ids`, or round-trip
decode output with `decoded`. Token IDs are model-specific, so keep fixture
files beside the model or dataset they validate.

5. Dry-run native training before writing adapter files:

```bash
go run ./cmd/nego train native ./models/qwen3 \
  --train-file examples/5.train/train.jsonl \
  --eval-file examples/5.train/test.jsonl \
  --dataset-format completion \
  --max-context 4096 \
  --dry-run
```

The dry run prints row counts, token budget, planned updated tokens, and top
tokens reinforced by the dataset.

6. Write a pure-Go native token-bias adapter:

```bash
go run ./cmd/nego train native ./models/qwen3 \
  --train-file examples/5.train/train.jsonl \
  --eval-file examples/5.train/test.jsonl \
  --dataset-format completion \
  --max-context 4096 \
  --out ./outputs/qwen3-token-bias
```

The output directory contains:

```text
adapter.json
manifest.json
README.md
```

7. Find the latest native adapter output:

```bash
go run ./cmd/nego train latest ./outputs --base-model ./models/qwen3
```

This prints the adapter output path plus ready-to-run `nego run` and
`nego chat` commands.

8. Chat with the trained adapter output:

```bash
go run ./cmd/nego chat ./outputs/qwen3-token-bias "Say hello in Indonesian." --max-tokens 16
```

9. Inspect and package the trained adapter:

```bash
go run ./cmd/nego inspect ./outputs/qwen3-token-bias
go run ./cmd/nego share manifest ./outputs/qwen3-token-bias --out ./outputs/qwen3-token-bias/share-manifest.json
go run ./cmd/nego share package ./outputs/qwen3-token-bias --out ./outputs/qwen3-token-bias.tar.gz
```

The current native trainer writes a token-bias adapter. Full pure-Go
LoRA/backprop training is planned after the native runtime matmul and optimizer
layers are ready.
