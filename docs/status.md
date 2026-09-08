# Nego Development Status

This page tracks what Nego can do today and what still blocks a pure-Go AI model development workflow.

## Usable Today

1. Download Hugging Face-style model files and GGUF runtime files.
2. Cache downloads and list, inspect, or remove local model registry entries.
3. Inspect model directories, model cards, config files, tokenizer files, safetensors metadata, GGUF metadata, and rough memory needs.
4. Load safetensors F32/F16/BF16 tensors into float32 buffers for native runtime development.
5. Map Hugging Face Qwen/Llama safetensors tensor names into native weight manifests.
6. Load Hugging Face safetensors directories through the experimental `native-hf` backend.
7. Load selected Hugging Face safetensors weights into float32 buffers for native runtime and training development.
8. Validate Hugging Face safetensors tensor shapes against local model config, including Qwen-style q/k attention norms.
9. Run native HF float32 math primitives for embedding lookup, projections, normalization, MLP, and logits.
10. Run guarded native HF generation with prompt decode state, KV cache history, adapter bias, and sampler controls for small compatible decoder fixtures.
11. Resolve local model artifacts into format, run backend, and training backend compatibility reports.
12. Tokenize text, count context usage, and render chat prompts with WordLevel, BPE, and Unigram/SentencePiece-style tokenizer metadata.
13. Run local GGUF chat through `llama.cpp` when `llama-cli` is installed.
14. Use remote OpenAI-compatible chat and embedding APIs.
15. Prepare, validate, convert, filter, split, and sample datasets.
16. Run eval suites and compare reports.
17. Create, validate, and run external training job JSON files.
18. Create local share manifests and archive packages for trained model output directories.
19. Upload regular files or small folders to Hub repos through inline commit uploads.

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
10. Early pure-Go token-bias adapter training for GGUF or Hugging Face safetensors downloads through `nego train native`.

Not production-ready yet:

1. Real Qwen/Llama compatibility for common downloaded GGUF models.
2. Compatibility validation for mixed K-quant variants such as Q4_K_M and Q5_K_M in real model files.
3. Optimized CPU execution, batching, and memory planning.
4. GPU execution.
5. Native safetensors autoregressive generation for production Qwen/Llama models.
6. Full GGUF LoRA/backprop training and optimizer support.

## Next Critical Phases

1. Add a native HF safetensors model loader that reuses the Qwen/Llama manifest.
2. Validate native forward math against known tiny Llama/Qwen GGUF fixtures.
3. Run K-quant compatibility tests against small real GGUF fixtures.
4. Add large-file Hub upload through LFS/Xet after auth, retry, and resumability are designed.
5. Design native training separately from runtime inference; keep external training orchestration stable until then.
