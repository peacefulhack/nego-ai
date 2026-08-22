# Nego Examples

Examples are numbered to show the usual AI model development flow.

1. [Download](1.download): download a model file with the Go hub API.
2. [Chat](2.chat): call an OpenAI-compatible chat backend and see a sample session file.
3. [Prepare Dataset](3.prepare-dataset): load CSV data and emit JSONL rows.
4. [Eval](4.eval): run an eval suite against a small in-memory example model.
5. [Train](5.train): read the `nego train` job JSON shape.
6. [Serve](6.serve): serve a local llama.cpp model through Nego HTTP.
7. [Embeddings](7.embeddings): call an embedding backend and compare vectors.
8. [Share Trained Model](8.share-trained-model): inspect a model card template and sharing checklist.

Run examples from the repository root:

```bash
go run ./examples/3.prepare-dataset
go run ./examples/4.eval
go run ./examples/5.train
go run ./examples/8.share-trained-model
```

Examples that call real model backends need local model files or backend environment variables.

For local GGUF models, GPU acceleration is controlled by runtime flags such as:

```bash
nego run ./models/model.gguf "Hello" --gpu full --flash-attn
nego chat ./models/model.gguf "Hello" --gpu off
nego serve ./models/model.gguf --gpu full --main-gpu 0 --tensor-split 3,1
```
