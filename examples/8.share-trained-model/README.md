# Step 8: Share the Trained Adapter

Run from the repository root. This example uses the actual pure-Go token-bias
adapter created in step 5. It creates a local archive; it does not upload files.

1. Download the base model and create the adapter using the included JSONL data:

```bash
go run ./cmd/nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3
go run ./examples/5.train/native_adapter
```

The native training example reads `examples/5.train/train.jsonl` and
`examples/5.train/test.jsonl`. Training rows use this format:

```json
{"prompt":"What is Go?","completion":"Go is a programming language."}
```

This adjusts token biases. It does not update transformer weights or train LoRA.

2. Inspect the output and package it:

```bash
go run ./cmd/nego inspect ./outputs/qwen3-token-bias
go run ./examples/8.share-trained-model
```

The archive is `outputs/qwen3-token-bias.tar.gz`. It includes `adapter.json`,
the training `manifest.json`, the generated `README.md`, and a
`nego-share-manifest.json` containing file hashes. Base weights are not bundled.
Review the generated model card before distributing it; the template in this
folder provides a starting point for additional model documentation.

An existing archive is not overwritten. To package another run, pass its output
directory; the archive is written beside that directory:

```bash
go run ./examples/8.share-trained-model ./outputs/my-adapter
```

3. Use the extracted adapter with the same base model on another machine:

```bash
go run ./cmd/nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3
go run ./cmd/nego chat ./models/qwen3 "What is Go?" --native --adapter ./outputs/qwen3-token-bias/adapter.json --max-tokens 64
```

Extract the archive into `outputs/qwen3-token-bias` first. Supplying the base
model and adapter separately avoids depending on the original machine's
absolute base-model path in the training manifest. Use the same base revision
and tokenizer used during training.
