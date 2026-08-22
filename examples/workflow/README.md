# Nego End-to-End Workflow Example

This folder contains a small runnable workflow example plus supporting files.

Run the Go example from the repository root:

```bash
go run ./examples/workflow
```

The example code reads the local CSV, eval suite, train job, and chat session files in this folder. It does not download or run a real model, so it is safe to run as a quick documentation smoke test.

## Numbered Steps

1. **Download a base model**

   ```bash
   nego download Qwen/Qwen3-0.6B --local-dir ./models/qwen3
   ```

2. **Inspect the downloaded model**

   ```bash
   nego inspect ./models/qwen3
   nego check ./models/qwen3
   ```

3. **Prepare a dataset**

   Input file:

   ```text
   examples/workflow/completion-data.csv
   ```

   Convert CSV to JSONL:

   ```bash
   nego dataset convert examples/workflow/completion-data.csv \
     --out data/completion.jsonl \
     --select prompt,completion \
     --require prompt,completion
   ```

4. **Validate and split the dataset**

   ```bash
   nego dataset validate data/completion.jsonl --format completion
   nego dataset split data/completion.jsonl --train-out data/train.jsonl --test-out data/test.jsonl
   ```

5. **Check token count and context budget**

   ```bash
   nego tokens ./models/qwen3 "Explain Go in simple terms."
   nego context ./models/qwen3 "Explain Go in simple terms." --max-context 4096
   ```

6. **Render a chat prompt**

   ```bash
   nego prompt ./models/qwen3 --system "You are helpful." --user "Explain Go."
   ```

7. **Chat with the model**

   ```bash
   nego chat ./models/model.gguf "Hello"
   nego chat ./models/model.gguf --interactive --session chats/qwen.json
   ```

   Example session file:

   ```text
   examples/workflow/chat-session.json
   ```

8. **Run a baseline eval**

   Eval suite:

   ```text
   examples/workflow/eval-suite.json
   ```

   Command:

   ```bash
   nego eval examples/workflow/eval-suite.json --json > reports/baseline.json
   nego eval report reports/baseline.json
   ```

9. **Train or fine-tune**

   Training job:

   ```text
   examples/workflow/train-job.json
   ```

   Command:

   ```bash
   nego train examples/workflow/train-job.json
   ```

10. **Inspect and evaluate the trained output**

    ```bash
    nego inspect ./outputs/qwen3-sft
    nego check ./outputs/qwen3-sft
    nego eval compare reports/baseline.json reports/trained.json
    ```

11. **Convert or optimize**

    ```bash
    nego convert gguf ./outputs/qwen3-sft \
      --out ./outputs/qwen3-sft.gguf \
      --converter /path/to/convert_hf_to_gguf.py
    ```

12. **Serve and share**

    ```bash
    nego serve ./outputs/qwen3-sft.gguf --addr :8080
    ```

    Direct package/upload commands are planned. For now, prepare a model directory with model files and a `README.md` model card.
