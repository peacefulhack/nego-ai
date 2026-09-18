# Pure-Go Linear LoRA

`adapters.LinearLoRA` implements a low-rank residual for a frozen linear layer:

```text
y = frozenOutput + (alpha / rank) * B * A * input
A: [rank, input_size]
B: [output_size, rank]
```

The API provides seeded initialization, forward evaluation, analytic gradients,
SGD updates, and Nego-format checkpoint loading/saving. A is normally initialized
and B starts at zero, so a fresh adapter does not change the base output.

## Training a Linear Adapter

```go
adapter, err := adapters.NewLinearLoRA(inputSize, outputSize, rank, alpha, seed)
if err != nil {
    return err
}
output, err := adapter.Forward(input, frozenOutput)
if err != nil {
    return err
}
// Compute dLoss/dOutput with the loss appropriate for the application.
gradient, err := adapter.Backward(input, outputGradient)
if err != nil {
    return err
}
if err := adapter.StepSGD(gradient, learningRate); err != nil {
    return err
}
return adapters.SaveLinearLoRA("./outputs/linear-step-1.json", adapter)
```

All vectors use float32; intermediate dot products accumulate in float64. Both
matrices are validated before a weight update is committed. Save uses a new
file only and refuses existing paths. Load has a 128 MiB file limit, while the
adapter has an eight-million-parameter safety cap. Save/load files are Nego JSON,
not PEFT safetensors or GGUF adapter files.

`gradient.Input` covers only the adapter branch. Backpropagating through a full
linear layer also requires the frozen projection's input gradient. Forward and
Backward must use the same weights; do not update between them. Calls that mutate
the adapter require exclusive access. Batching, gradient clipping, and loss
reduction are the caller's responsibility.

## Validation

```powershell
go test ./adapters -run LinearLoRA -count=1
```

Tests compare every A, B, and input derivative with central finite differences,
check that SGD learns a small regression task, verify checkpoint resume, and
reject malformed dimensions, non-finite updates, and oversized files.

## Integration Status

The GGUF backend now exposes `ForwardFeaturesWithState` for collecting frozen
final hidden states and base logits. `training.FitLinearLoRA` trains an output-head
adapter from those samples with next-token cross-entropy, SGD, and optional L2
gradient clipping. `training.EvaluateLinearLoRA` evaluates a separate sample set
without updates. Training supports cancellation between samples and retains
completed updates on cancellation.

Load a saved output-head checkpoint with the native backend option `output_lora`.
The loader checks embedding/vocabulary dimensions; it does not fingerprint the
base model. Always reuse the original base model and tokenizer. Feature extraction
deliberately bypasses attached output adapters so features remain frozen-base data.

Run the [step-5 output LoRA example](../examples/5.train/output_lora) for a complete
JSONL-to-checkpoint-to-chat workflow using the downloaded Qwen3 GGUF. A local
Q4_K_M smoke run trained six completion targets for three epochs and reduced
training loss from 4.9817 to 4.7919, then reloaded the checkpoint for chat. This is
a pipeline check on training data, not evidence of improved held-out quality.

`nego train native` still trains token-bias adapters. Full attention-layer LoRA,
transformer backprop, native-HF integration, Adam optimizer state, streaming
datasets, and portable adapter manifests are not implemented by this API yet.
No GPU execution is provided.

Reference: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
