# Step 5: Output-Head LoRA on the Downloaded GGUF

Run from the repository root. The model comes from the GGUF download step, not a
remote inference API. Python and llama.cpp executables are not required.

1. Download the tested base model:

```powershell
go run ./cmd/nego download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf --local-dir ./models/qwen3-gguf
```

2. Read or edit `train.jsonl` in this folder. Each line contains a `prompt` and
   `completion` string. The example inserts a newline between them, masks prompt
   targets, and rejects ambiguous tokenization boundaries. Data is embedded when
   the example is built, so edits take effect on the next `go run`.

3. Train, save, reload, and chat:

```powershell
go run ./examples/5.train/output_lora
```

4. For a new experiment, choose another checkpoint filename:

```powershell
go run ./examples/5.train/output_lora -epochs 5 -out ./outputs/qwen3-output-lora-v2.json
```

5. Load the checkpoint in your application with the SAME base model:

```go
model, err := nego.LoadModel(ctx, nego.ModelOptions{
    Backend: native.BackendName,
    Path: "./models/qwen3-gguf",
    Options: map[string]string{"output_lora": "./outputs/qwen3-output-lora.json"},
})
```

Import `github.com/gakon/nego-ai/backends/native` to register that backend. The
checkpoint has shape checks but no base-model fingerprint; do not attach it to a
different model just because dimensions match.

This trains real low-rank A/B weights on the output projection using cross-entropy
and clipped SGD. Transformer weights remain frozen; it is not attention-layer
LoRA, full fine-tuning, or a PEFT/GGUF adapter export. Reported loss is on the tiny
training data, not validation accuracy. The example limits cached features to 128
completion targets; large datasets need a streaming training pipeline. CPU RAM
usage still includes the approximately 2.2 GiB decoded Qwen3 base model.
