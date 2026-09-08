# Nego Examples

Examples are numbered to show the usual AI model development flow.

1. [Download](1.download): download the Hugging Face base model to `./models/qwen3` and a GGUF runtime model to `./models/qwen3-gguf`.
2. [Chat](2.chat): chat with the Hugging Face safetensors model downloaded in step 1 through the pure-Go native-HF backend.
3. [Prepare Dataset](3.prepare-dataset): load CSV data and emit JSONL rows.
4. [Eval](4.eval): define an eval suite for the GGUF model downloaded in step 1; the Go example runs the same cases against a tiny in-memory model.
5. [Train](5.train): create a LoRA training job for the Hugging Face model and a native token-bias adapter from a downloaded model.
6. [Serve](6.serve): serve a local llama.cpp model through Nego HTTP.
7. [Embeddings](7.embeddings): call an embedding backend and compare vectors.
8. [Share Trained Model](8.share-trained-model): inspect a model card template and sharing checklist.

Run examples from the repository root:

```bash
go run ./examples/1.download
go run ./examples/2.chat
go run ./examples/3.prepare-dataset
go run ./examples/4.eval
go run ./examples/5.train
go run ./examples/5.train/native_adapter
go run ./examples/8.share-trained-model
```

The default chat example uses Nego's experimental pure-Go local backend. Examples that explicitly use `llama.cpp` need `llama-cli` on PATH or `NEGO_LLAMA_CLI` set.

For local GGUF models, GPU acceleration is controlled by runtime flags such as:

```bash
nego run --backend llama.cpp ./models/qwen3-gguf "Hello" --gpu full --flash-attn
nego chat --backend llama.cpp ./models/qwen3-gguf "Hello" --gpu off
nego serve ./models/qwen3-gguf --gpu full --main-gpu 0 --tensor-split 3,1
```
