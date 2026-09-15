# Nego Development Status

This page tracks what Nego can do today and what still blocks a pure-Go AI model development workflow.

## Usable Today

1. Download Hugging Face-style model files and GGUF runtime files.
2. Cache downloads and list, inspect, or remove local model registry entries.
3. Inspect model directories, model cards, config files, generation defaults, tokenizer files, safetensors metadata, GGUF metadata, backend readiness, and rough memory needs.
4. Load safetensors F32/F16/BF16 tensors into float32 buffers for native runtime development.
5. Map Hugging Face Qwen/Llama safetensors tensor names into native weight manifests.
6. Load Hugging Face safetensors directories through the experimental `native-hf` backend.
7. Load selected Hugging Face safetensors weights into float32 buffers for native runtime and training development.
8. Validate Hugging Face safetensors tensor shapes against local model config, including Qwen-style q/k attention norms.
9. Run native HF float32 math primitives for embedding lookup, projections, normalization, MLP, and logits.
10. Run experimental native HF generation, streaming completions, streaming chat, and local serving with prompt decode state, KV cache history, adapter bias, sampler controls, context guards, and local generation defaults.
11. Resolve local model artifacts into format, run backend, training backend, and native GGUF readiness reports.
12. Inspect and check native training output directories as runnable adapter artifacts.
13. Tokenize text, count context usage, and render chat prompts with WordLevel, BPE, and Unigram/SentencePiece-style tokenizer metadata.
14. Run local GGUF chat through `llama.cpp` when `llama-cli` is installed.
15. Use remote OpenAI-compatible chat and embedding APIs.
16. Prepare, quality-check, validate, token-budget check, render, convert, filter, dedupe, split, and sample datasets.
17. Export reusable run/chat config files from local model or adapter paths.
18. Log run/chat and native training experiments, inspect them, and compare two logged runs.
19. Run eval suites and compare reports.
20. Create, validate, and run external training job JSON files.
21. Create local native training manifests with provenance, share manifests, upload preflight checks, and archive packages for trained model output directories.
22. Upload regular files or small folders to Hub repos through inline commit uploads.

## Pure-Go Native Runtime Status

The `native` backend is experimental and does not require `llama-cli`.

Implemented:

1. GGUF metadata, tensor directory, and tokenizer vocabulary loading.
2. GGUF vocabulary encode/decode with BPE merge metadata support.
3. Tensor loading into float32 buffers with per-model caching.
4. F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K dequantization.
5. RMSNorm, SiLU, softmax, vector math, RoPE, attention, KV-cache, MLP, logits, and transformer block primitives.
6. Qwen-style optional Q/K attention norm tensors in native GGUF forward passes.
7. Decode state plumbing that processes prompt tokens and appends K/V vectors.
8. Early `Generate` and `Chat` loops for supported tiny GGUF fixtures.
9. EOS stopping, context guards, repeat penalty controls, streaming completions, and streaming chat for native sampling.
10. CLI access through `nego run --native` and `nego chat --native`.
11. Early pure-Go token-bias adapter training for GGUF or Hugging Face safetensors downloads through `nego train native`.
12. Native adapter manifests with training provenance, reusable commands, and direct load support through output directories.
13. Actionable native GGUF runtime errors that point to missing tensors, shape mismatches, and `nego check` diagnostics.
14. `--native` CLI runtime selection for GGUF (`native`) and Hugging Face safetensors (`native-hf`) pure-Go backends.
15. Native-HF load preflight for incomplete config, missing tensors, and shape mismatches before generation starts.
16. Public `nego.BackendPureGo` / `nego.BackendNativeAuto` aliases for requiring local pure-Go runtime resolution from Go code.
17. Native-HF float32 tensor caching per loaded model instance for faster repeated forward steps.
18. Native GGUF load preflight for incomplete manifests and unsupported required tensor types before opening tensor storage.
19. Native readiness reports name unsupported required or present optional GGUF tensors, not only their tensor type family.
20. Native GGUF tensor caching can be disabled per model with `cache_tensors=false` for lower-memory inspection experiments.
21. Memory estimates include approximate decoded float32 tensor cache needs for native HF and GGUF runtime experiments.
22. Native token-bias training reports a warning when the base model is trainable but its local generation backend is not ready.
23. Native training output manifests and generated READMEs persist runtime-readiness warnings.
24. Runtime CLI commands expose `--no-cache-tensors` as a direct lower-memory native backend option.
25. Runtime CLI commands print model-loading progress before potentially slow backend initialization.
26. Native GGUF raw tensor reads use a configurable `max_tensor_read_bytes` safety limit.
27. Native GGUF spec, readiness, memory, and manifest checks honor `attention.key_length` / `attention.value_length` metadata for Qwen-style projection dimensions.
28. Pure-Go runtime backends can cap decoded float32 tensor cache memory with `max_tensor_cache_bytes` / `--max-tensor-cache-bytes`.
29. `nego check`, `modelinfo.Check`, and `nego train capabilities` include a quick local memory summary when model dimensions are available.
30. Native training results, adapter manifests, and generated adapter READMEs persist the base model memory estimate when available.
31. Native GGUF prompt and training tokenization honor `tokenizer.ggml.add_bos_token` / `add_eos_token` defaults.
32. GGUF tokenizer inspection reads pre-tokenizer, whitespace, EOT, and EOM metadata used by common converted model files.
33. `nego check` reports GGUF tokenizer model, pre-tokenizer, vocab size, special token IDs, and tokenizer defaults.
34. Native GGUF readiness warns when parsed pre-tokenizer metadata may still need token-ID validation before training or output comparisons.
35. `nego tokenize`, `nego tokens`, and `nego context` can use GGUF vocabulary metadata when `tokenizer.json` is not present.
36. `nego tokenize check` validates tokenization fixtures with expected token IDs and decoded text for local tokenizer regression checks.
37. GGUF decode skip-special handling recognizes common special token strings even when the special token ID is zero.
38. Dataset token-budget checks can use an injected token counter, and the CLI uses that path for GGUF vocabulary fallback.
39. `nego train capabilities` includes tokenizer diagnostics when the inspected model exposes GGUF tokenizer metadata.
40. `nego train native` prints a follow-up share packaging command after writing a native adapter artifact.
41. Native training propagates backend readiness warnings, including tokenizer validation warnings, into training results and adapter manifests.
42. The root `nego` package exposes convenience helpers for model inspection, artifact resolution, training assessment, and native training.
43. Native training results, adapter manifests, and generated adapter READMEs persist GGUF tokenizer metadata when available.
44. `nego train native --tokenize-check` validates tokenizer fixtures before training starts.

Not production-ready yet:

1. Full real-world Qwen/Llama compatibility validation for common downloaded GGUF models.
2. Compatibility validation for mixed K-quant variants such as Q4_K_M and Q5_K_M in real model files.
3. Optimized CPU execution, batching, and memory planning.
4. GPU execution.
5. Production-quality native safetensors autoregressive generation for Qwen/Llama models.
6. Full GGUF LoRA/backprop training and optimizer support.

## Next Critical Phases

1. Add curated Llama/Qwen tokenizer fixture packs for `nego tokenize check`.
2. Validate native forward math against known tiny Llama/Qwen GGUF and safetensors fixtures.
3. Run K-quant compatibility tests against small real GGUF fixtures.
4. Add optimized CPU execution, batching, and memory reuse for larger local models.
5. Add GPU execution behind a clean backend interface once CPU correctness is stable.
6. Add full LoRA/backprop training and optimizer checkpoints for GGUF or safetensors artifacts.
7. Add large-file Hub upload through LFS/Xet after auth, retry, and resumability are designed.
