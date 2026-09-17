package cli

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

func TestStatusRuntimeSmokeSuccess(t *testing.T) {
	path := fakeNativeHFModel(t)
	// Use a valid tokenizer with the existing tiny, zero-weight safetensors model.
	if err := os.WriteFile(filepath.Join(path, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":0,"world":1,"test":2,"<unk>":3},"unk_token":"<unk>"},"pre_tokenizer":{"type":"Whitespace"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, jsonOutput := range []bool{false, true} {
		t.Run(map[bool]string{false: "text", true: "json"}[jsonOutput], func(t *testing.T) {
			// Boolean flags must work before the path as well as after it.
			args := []string{"status", "--runtime-smoke", "--runtime-stats", path, "--smoke-prompt", "world"}
			if jsonOutput {
				args = append(args, "--json")
			}
			var stdout, stderr bytes.Buffer
			if code := Run(context.Background(), args, &stdout, &stderr); code != 0 {
				t.Fatalf("code=%d stdout=%s stderr=%s", code, &stdout, &stderr)
			}
			if !jsonOutput {
				for _, want := range []string{"Runtime smoke:", "Passed:  yes", `Prompt:  "world"`, `Output:  "hello"`, "Runtime stats:"} {
					if !strings.Contains(stdout.String(), want) {
						t.Fatalf("missing %q in %s", want, &stdout)
					}
				}
				if !strings.Contains(stderr.String(), "Running runtime smoke test") {
					t.Fatalf("missing progress: %s", &stderr)
				}
				return
			}
			var body struct {
				Artifact     *modelinfo.Artifact `json:"artifact"`
				RuntimeSmoke statusRuntimeSmoke  `json:"runtime_smoke"`
				RuntimeStats *nego.RuntimeStats  `json:"runtime_stats"`
			}
			if err := json.Unmarshal(stdout.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body.Artifact == nil || !body.RuntimeSmoke.Passed || body.RuntimeSmoke.Backend != "native-hf" || body.RuntimeSmoke.Output != "hello" || body.RuntimeSmoke.Prompt != "world" {
				t.Fatalf("unexpected report: %s", &stdout)
			}
			if body.RuntimeStats == nil || body.RuntimeStats.CachedTensors == 0 {
				t.Fatalf("expected stats from generation, got %s", &stdout)
			}
			if stderr.Len() != 0 {
				t.Fatalf("JSON mode wrote stderr: %s", &stderr)
			}
		})
	}
}

func TestStatusRuntimeSmokeFailure(t *testing.T) {
	for _, format := range []string{"gguf", "hf"} {
		t.Run(format, func(t *testing.T) {
			path := fakeInspectGGUF(t)
			if format == "hf" {
				path = fakeNativeHFModel(t)
			}
			for _, jsonOutput := range []bool{false, true} {
				var stdout, stderr bytes.Buffer
				args := []string{"status", path, "--runtime-smoke"}
				if jsonOutput {
					args = append(args, "--json")
				}
				if code := Run(context.Background(), args, &stdout, &stderr); code != 1 {
					t.Fatalf("code=%d stdout=%s stderr=%s", code, &stdout, &stderr)
				}
				if jsonOutput {
					var body struct {
						Artifact     *modelinfo.Artifact `json:"artifact"`
						RuntimeSmoke *statusRuntimeSmoke `json:"runtime_smoke"`
						RuntimeStats *nego.RuntimeStats  `json:"runtime_stats"`
					}
					if err := json.Unmarshal(stdout.Bytes(), &body); err != nil {
						t.Fatal(err)
					}
					if body.Artifact == nil || body.RuntimeSmoke == nil || body.RuntimeSmoke.Passed || body.RuntimeSmoke.Error == "" || body.RuntimeStats != nil {
						t.Fatalf("unexpected failure report: %s", &stdout)
					}
					if stderr.Len() != 0 {
						t.Fatalf("JSON mode wrote stderr: %s", &stderr)
					}
				} else {
					for _, want := range []string{"Format:", "Runtime smoke:", "Passed:  no", "Error:"} {
						if !strings.Contains(stdout.String(), want) {
							t.Fatalf("missing %q in %s", want, &stdout)
						}
					}
					if !strings.Contains(stderr.String(), "runtime smoke failed") {
						t.Fatalf("missing failure message: %s", &stderr)
					}
				}
			}
		})
	}
}

func TestStatusRuntimeSmokeGGUFSuccess(t *testing.T) {
	path := fakeSmokeGGUF(t)
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), []string{"status", path, "--runtime-smoke", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, &stdout, &stderr)
	}
	var body struct {
		RuntimeSmoke statusRuntimeSmoke `json:"runtime_smoke"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if !body.RuntimeSmoke.Passed || body.RuntimeSmoke.Backend != "native" || body.RuntimeSmoke.Output != "hello" {
		t.Fatalf("unexpected GGUF result: %s", &stdout)
	}
}

func fakeSmokeGGUF(t *testing.T) string {
	t.Helper()
	tensors := []struct {
		name  string
		shape []uint64
	}{
		{"token_embd.weight", []uint64{4, 2}},
		{"output_norm.weight", []uint64{4}},
		{"blk.0.attn_norm.weight", []uint64{4}},
		{"blk.0.attn_q.weight", []uint64{4, 4}},
		{"blk.0.attn_k.weight", []uint64{4, 2}},
		{"blk.0.attn_v.weight", []uint64{4, 2}},
		{"blk.0.attn_output.weight", []uint64{4, 4}},
		{"blk.0.ffn_norm.weight", []uint64{4}},
		{"blk.0.ffn_gate.weight", []uint64{4, 8}},
		{"blk.0.ffn_up.weight", []uint64{4, 8}},
		{"blk.0.ffn_down.weight", []uint64{8, 4}},
	}
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	for _, value := range []any{uint32(3), uint64(len(tensors)), uint64(9)} {
		if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeInspectGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeInspectGGUFUint32KV(t, &buf, "llama.context_length", 8)
	writeInspectGGUFUint32KV(t, &buf, "llama.embedding_length", 4)
	writeInspectGGUFUint32KV(t, &buf, "llama.block_count", 1)
	writeInspectGGUFUint32KV(t, &buf, "llama.feed_forward_length", 8)
	writeInspectGGUFUint32KV(t, &buf, "llama.attention.head_count", 2)
	writeInspectGGUFUint32KV(t, &buf, "llama.attention.head_count_kv", 1)
	writeInspectGGUFStringKV(t, &buf, "tokenizer.ggml.model", "gpt2")
	writeInspectGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"hello", "world"})
	var dataBytes uint64
	for _, tensor := range tensors {
		writeInspectGGUFTensor(t, &buf, tensor.name, tensor.shape, 0, dataBytes)
		size := uint64(4)
		for _, dim := range tensor.shape {
			size *= dim
		}
		dataBytes += (size + 31) / 32 * 32
	}
	buf.Write(make([]byte, (32-buf.Len()%32)%32))
	buf.Write(make([]byte, dataBytes))
	path := filepath.Join(t.TempDir(), "smoke.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestStatusRuntimeSmokeCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var stdout, stderr bytes.Buffer
	code := Run(ctx, []string{"status", fakeInspectGGUF(t), "--runtime-smoke", "--json"}, &stdout, &stderr)
	if code != 1 || !strings.Contains(stdout.String(), "context canceled") {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, &stdout, &stderr)
	}
}

func TestStatusRuntimeSmokeInvalidFlags(t *testing.T) {
	for _, args := range [][]string{
		{"status", "unused", "--smoke-prompt", "hello"},
		{"status", "unused", "--runtime-smoke", "--smoke-prompt", " "},
	} {
		var stdout, stderr bytes.Buffer
		if code := Run(context.Background(), args, &stdout, &stderr); code != 2 {
			t.Fatalf("args=%v code=%d stdout=%s stderr=%s", args, code, &stdout, &stderr)
		}
	}
}
