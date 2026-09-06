# Nego Development Status

This page tracks what Nego can do today and what still blocks a pure-Go AI model development workflow.

## Usable Today

1. Download Hugging Face-style model files and GGUF runtime files.
2. Cache downloads and list, inspect, or remove local model registry entries.
3. Inspect model directories, model cards, config files, tokenizer files, safetensors metadata, and GGUF metadata.
4. Resolve local model artifacts into format, run backend, and training backend compatibility reports.
5. Tokenize text, count context usage, and render chat prompts with WordLevel, BPE, and Unigram/SentencePiece-style tokenizer metadata.
6. Run local GGUF chat through `llama.cpp` when `llama-cli` is installed.
7. Use remote OpenAI-compatible chat and embedding APIs.
8. Prepare, validate, convert, filter, split, and sample datasets.
9. Run eval suites and compare reports.
10. Create, validate, and run external training job JSON files.
11. Create local share manifests and archive packages for trained model output directories.
12. Upload regular files or small folders to Hub repos through inline commit uploads.

## Pure-Go Native Runtime Status

The `native` backend is experimental and does not require `llama-cli`.

Implemented:

1. GGUF metadata, tensor directory, and tokenizer vocabulary loading.
2. GGUF vocabulary encode/decode with BPE merge metadata support.
3. Tensor loading into float32 buffers with per-model caching.
4. F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K dequantization.
5. RMSNorm, SiLU, softmax, vector math, RoPE, attention, KV-cache, MLP, logits, and transformer block primitives.
6. Decode state plumbing that processes prompt tokens and appends K/V vectors.
7. Early `Generate` and `Chat` loops for supported tiny GGUF fixtures.
8. EOS stopping, repeat penalty controls, and streaming chat for native sampling.
9. CLI access through `nego run --native` and `nego chat --native`.

Not production-ready yet:

1. Real Qwen/Llama compatibility for common downloaded GGUF models.
2. Compatibility validation for mixed K-quant variants such as Q4_K_M and Q5_K_M in real model files.
3. Optimized CPU execution, batching, and memory planning.
4. GPU execution.
5. Native safetensors forward-pass loading and fine-tuning/training.

## Next Critical Phases

1. Add native safetensors tensor reading into typed buffers.
2. Validate native forward math against known tiny Llama/Qwen GGUF fixtures.
3. Run K-quant compatibility tests against small real GGUF fixtures.
4. Add large-file Hub upload through LFS/Xet after auth, retry, and resumability are designed.
5. Design native training separately from runtime inference; keep external training orchestration stable until then.
