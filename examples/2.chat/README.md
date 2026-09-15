# Step 2: Chat With The Downloaded Model

This step uses the model from step 1. It does not call a remote OpenAI-compatible
API.

1. Download a local model first:

```bash
go run ./cmd/nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3
```

2. Check which local runtime Nego can use:

```bash
go run ./cmd/nego check ./models/qwen3
```

3. Run the Go example against the downloaded path:

```bash
go run ./examples/2.chat ./models/qwen3
```

4. Try a GGUF download when you want the `native` GGUF path instead:

```bash
go run ./cmd/nego download Qwen/Qwen3-0.6B --gguf --local-dir ./models/qwen3-gguf
go run ./cmd/nego check ./models/qwen3-gguf
go run ./examples/2.chat ./models/qwen3-gguf
```

5. If the model is not ready for pure-Go generation yet, inspect the blocker:

```bash
go run ./cmd/nego status ./models/qwen3
go run ./cmd/nego inspect ./models/qwen3 --json
```

The example uses:

```go
model, err := nego.LoadModel(context.Background(), nego.ModelOptions{
    Backend: nego.BackendPureGo,
    Path:    path,
})
```

`BackendPureGo` requires a local Nego native backend. If no pure-Go backend is
ready for that downloaded model yet, Nego returns an error instead of silently
calling a remote service.
