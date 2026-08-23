# Nego Development Status

This page tracks what Nego can do today and what still blocks a pure-Go AI model development workflow.

## Usable Today

1. Download Hugging Face-style model files and GGUF runtime files.
2. Cache downloads and list, inspect, or remove local model registry entries.
3. Inspect model directories, model cards, config files, tokenizer files, and GGUF metadata.
4. Tokenize text, count context usage, and render chat prompts.
5. Run local GGUF chat through `llama.cpp` when `llama-cli` is installed.
6. Use remote OpenAI-compatible chat and embedding APIs.
7. Prepare, validate, convert, filter, split, and sample datasets.
8. Run eval suites and compare reports.
9. Create, validate, and run external training job JSON files.
10. Create local share manifests for trained model output directories.

## Pure-Go Native Runtime Status

The `native` backend is experimental and does not require `llama-cli`.

Implemented:

1. GGUF metadata, tensor directory, and tokenizer vocabulary loading.
2. Basic GGUF vocabulary encode/decode.
3. Tensor loading into float32 buffers with per-model caching.
4. F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and Q8_1 dequantization.
5. RMSNorm, SiLU, softmax, vector math, RoPE, attention, KV-cache, MLP, logits, and transformer block primitives.
6. Decode state plumbing that processes prompt tokens and appends K/V vectors.
7. Early `Generate` and `Chat` loops for supported tiny GGUF fixtures.
8. CLI access through `nego run --native` and `nego chat --native`.

Not production-ready yet:

1. Real Qwen/Llama compatibility for common downloaded GGUF models.
2. K-quant kernels such as Q4_K_M, Q5_K_M, and Q6_K.
3. Optimized CPU execution, batching, and memory planning.
4. GPU execution.
5. Native fine-tuning/training.

## Next Critical Phases

1. Add K-quant tensor kernels and compatibility tests with small fixture blocks.
2. Validate native forward math against known tiny Llama/Qwen GGUF fixtures.
3. Add tokenizer coverage for real SentencePiece/BPE metadata variants.
4. Add native generation quality controls: repeat penalty, EOS handling, and streaming.
5. Add model packaging archive support on top of share manifests.
6. Add hub upload after auth, retry, and large-file handling are designed.
7. Design native training separately from runtime inference; keep external training orchestration stable until then.
