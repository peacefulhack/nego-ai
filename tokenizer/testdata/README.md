# Qwen Tokenizer Reference

`qwen_reference.json` contains 24 cases produced independently by Hugging Face
`tokenizers==0.21.1` using a local `Qwen/Qwen3-0.6B/tokenizer.json`. The fixture
records that file's SHA-256, token IDs, decoded text, and raw pre-tokenizer splits.
Inputs cover multilingual text, NFC accents, whitespace, numbers, contractions,
code, and adjacent chat special tokens.

Normal `go test ./...` checks the splitter against these reference segments and
exercises the complete encoder with a small synthetic vocabulary. It requires
neither Python nor a downloaded model.

To also compare complete token IDs and decoded text against the real vocabulary,
set an absolute path to your local tokenizer before running Go tests:

```powershell
$env:NEGO_QWEN_TOKENIZER_DIR = (Resolve-Path ./models/qwen3).Path
go test ./tokenizer -run Qwen -count=1
```

The tests never download a model. Use the recorded tokenizer file when checking
these golden IDs; other models or revisions may have different vocabularies.

To regenerate the fixture, install `tokenizers==0.21.1` in a separate development
environment and run from the repository root:

```bash
python tokenizer/testdata/generate_qwen_reference.py models/qwen3 tokenizer/testdata/qwen_reference.json
```

Python is only an optional reference-fixture generator, not a Nego dependency.
Review changed reference IDs against the source tokenizer before committing.

Reference algorithm: [Hugging Face Qwen2 tokenizer](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/tokenization_qwen2.py).

## GGUF Qwen Profile

`qwen_gguf_reference.json` uses the same vocabulary and 24 inputs, with the HF
normalizer disabled. GGUF `gpt2`/`qwen2` metadata identifies the byte-level split
and BPE pipeline but does not declare HF NFC normalization. Consequently,
decomposed accents retain their original bytes in this profile.

```powershell
$env:NEGO_QWEN_GGUF = (Resolve-Path ./models/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf).Path
go test ./modelinfo -run QwenGGUFLocalReference -count=1
```

This opt-in check compares exact IDs, batch decoding, and Unicode-safe streaming
against the independent reference. It neither downloads nor loads model weights.
The tested GGUF vocabulary is from `unsloth/Qwen3-0.6B-GGUF`.

Regenerate only in the optional reference environment:

```bash
python tokenizer/testdata/generate_qwen_reference.py models/qwen3 tokenizer/testdata/qwen_gguf_reference.json --no-normalizer
```
