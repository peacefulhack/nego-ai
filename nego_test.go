package nego

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
)

func TestLoadModelUsesRegisteredBackend(t *testing.T) {
	name := testBackendName("mock")
	if err := RegisterBackend(name, mockBackend{}); err != nil {
		t.Fatal(err)
	}
	model, err := LoadModel(context.Background(), ModelOptions{Backend: name, Path: "./model"})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	out, err := model.Generate(context.Background(), GenerateRequest{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if out.Text != "generated: hello" {
		t.Fatalf("Generate = %#v", out)
	}

	chat, err := model.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hello"}}})
	if err != nil {
		t.Fatal(err)
	}
	if chat.Message.Role != RoleAssistant || chat.Message.Content != "chat: hello" {
		t.Fatalf("Chat = %#v", chat)
	}

	stream, err := model.StreamChat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hello"}}})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for token := range stream.Tokens() {
		text += token.Text
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if text != "stream" {
		t.Fatalf("stream text = %q", text)
	}
}

func TestEmbedUsesOptionalCapability(t *testing.T) {
	resp, err := Embed(context.Background(), embeddingModel{}, EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) != 2 {
		t.Fatalf("unexpected embeddings: %#v", resp)
	}
	if _, err := Embed(context.Background(), mockModel{}, EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("expected unsupported embedding error")
	}
}

func TestStreamGenerateFallsBackToGenerate(t *testing.T) {
	stream, err := StreamGenerate(context.Background(), mockModel{}, GenerateRequest{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for token := range stream.Tokens() {
		text += token.Text
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if text != "generated: hello" {
		t.Fatalf("stream text = %q", text)
	}
}

func TestLoadModelRequiresRegisteredBackend(t *testing.T) {
	if _, err := LoadModel(context.Background(), ModelOptions{Backend: "missing"}); err == nil {
		t.Fatal("expected missing backend error")
	}
}

func TestLoadModelAutoSelectsRemoteBackend(t *testing.T) {
	name := "openai-compatible"
	if _, ok := BackendInfoByName(name); !ok {
		if err := RegisterBackend(name, mockBackend{}); err != nil {
			t.Fatal(err)
		}
	}
	model, err := LoadModel(context.Background(), ModelOptions{
		Endpoint: "http://localhost:8080/v1",
		Model:    "local-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
}

func TestLoadModelPureGoAliasUsesResolvedBackend(t *testing.T) {
	if _, ok := BackendInfoByName("native"); !ok {
		if err := RegisterBackend("native", mockBackend{}); err != nil {
			t.Fatal(err)
		}
	}
	model, err := LoadModel(context.Background(), ModelOptions{Backend: BackendPureGo, Path: fakeNativeReadyGGUF(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
}

func TestResolvePureGoBackendRejectsRemoteBackend(t *testing.T) {
	_, err := ResolvePureGoBackend(ModelOptions{Backend: BackendNativeAuto, Endpoint: "http://localhost:8080/v1", Model: "remote"})
	if err == nil {
		t.Fatal("expected pure-Go remote rejection")
	}
}

func TestResolveBackendRequiresLocalOrRemoteTarget(t *testing.T) {
	if _, err := ResolveBackend(ModelOptions{}); err == nil {
		t.Fatal("expected target error")
	}
}

func TestBackendInfoDiscovery(t *testing.T) {
	name := testBackendName("described")
	if err := RegisterBackend(name, describedBackend{}); err != nil {
		t.Fatal(err)
	}
	info, ok := BackendInfoByName(name)
	if !ok {
		t.Fatal("expected backend info")
	}
	if info.Name != name || info.Description != "test backend" || len(info.Capabilities) != 1 {
		t.Fatalf("info = %#v", info)
	}
	found := false
	for _, backend := range ListBackends() {
		if backend.Name == name {
			found = true
		}
	}
	if !found {
		t.Fatalf("backend %q not found in list", name)
	}
}

func TestRootConvenienceAPIs(t *testing.T) {
	modelPath := fakeNativeReadyGGUF(t)
	info, err := Inspect(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.GGUF == nil {
		t.Fatalf("expected GGUF info: %#v", info)
	}
	check, err := Check(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if check.Artifact == nil || check.Artifact.Format != "gguf" {
		t.Fatalf("unexpected check: %#v", check)
	}
	artifact, err := ResolveArtifact(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.RecommendedRunBackend == "" {
		t.Fatalf("unexpected artifact: %#v", artifact)
	}
	assessment, err := AssessTraining(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(assessment.Methods) == 0 {
		t.Fatalf("unexpected assessment: %#v", assessment)
	}
	dir := t.TempDir()
	trainFile := filepath.Join(dir, "train.jsonl")
	if err := os.WriteFile(trainFile, []byte(`{"prompt":"hi","completion":"hello"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result, err := TrainNative(context.Background(), NativeTrainingOptions{
		BaseModel:     modelPath,
		TrainFile:     trainFile,
		DatasetFormat: "completion",
		OutputDir:     filepath.Join(dir, "adapter"),
		DryRun:        true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.DryRun || result.UpdatedTokens == 0 {
		t.Fatalf("unexpected training result: %#v", result)
	}
	adapterDir := filepath.Join(dir, "adapter-real")
	if err := os.MkdirAll(adapterDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := `{"version":1,"type":"nego-native-adapter","base_model":"` + filepath.ToSlash(modelPath) + `","adapter_path":"adapter.json","method":"token-bias","recommended_backend":"native","runtime_options":{"adapter_path":"adapter.json"},"created_at":"2026-01-01T00:00:00Z"}`
	if err := os.WriteFile(filepath.Join(adapterDir, "manifest.json"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	latest, err := LatestNativeTrainingManifest(NativeTrainingManifestDiscoveryOptions{Root: dir})
	if err != nil {
		t.Fatal(err)
	}
	if latest.Path != adapterDir || latest.Manifest.BaseModel == "" {
		t.Fatalf("unexpected latest manifest: %#v", latest)
	}

	tokDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tokDir, "tokenizer.json"), []byte(`{"model":{"type":"WordLevel","vocab":{"hello":1},"unk_token":"hello"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(tokDir)
	if err != nil {
		t.Fatal(err)
	}
	cases, err := ValidateTokenizerCheckCases([]TokenizerCheckCase{{Text: "hello", Tokens: []int{1}}})
	if err != nil {
		t.Fatal(err)
	}
	checkResult := CheckTokenizer(tok, cases)
	if checkResult.Passed != 1 || checkResult.Failed != 0 {
		t.Fatalf("unexpected tokenizer check: %#v", checkResult)
	}
}

var testBackendCounter int64

func testBackendName(prefix string) string {
	return prefix + "-test-backend-" + strconv.FormatInt(atomic.AddInt64(&testBackendCounter, 1), 10)
}

type mockBackend struct{}

func (mockBackend) Load(context.Context, ModelOptions) (Model, error) {
	return mockModel{}, nil
}

type mockModel struct{}

func (mockModel) Generate(_ context.Context, req GenerateRequest) (*GenerateOutput, error) {
	return &GenerateOutput{Text: "generated: " + req.Prompt}, nil
}

func (mockModel) Chat(_ context.Context, req ChatRequest) (*ChatResponse, error) {
	content := ""
	if len(req.Messages) > 0 {
		content = req.Messages[len(req.Messages)-1].Content
	}
	return &ChatResponse{Message: Message{Role: RoleAssistant, Content: "chat: " + content}}, nil
}

func (mockModel) StreamChat(context.Context, ChatRequest) (Stream, error) {
	ch := make(chan Token, 1)
	ch <- Token{Text: "stream"}
	close(ch)
	return mockStream{tokens: ch}, nil
}

func (mockModel) Close() error {
	return nil
}

type mockStream struct {
	tokens <-chan Token
}

func (s mockStream) Tokens() <-chan Token {
	return s.tokens
}

func (mockStream) Err() error {
	return nil
}

func (mockStream) Close() error {
	return nil
}

type embeddingModel struct {
	mockModel
}

func (embeddingModel) Embed(context.Context, EmbeddingRequest) (*EmbeddingResponse, error) {
	return &EmbeddingResponse{Embeddings: [][]float64{{1, 0}}}, nil
}

type describedBackend struct {
	mockBackend
}

func (describedBackend) Info() BackendInfo {
	return BackendInfo{
		Description:  "test backend",
		Capabilities: []string{"generate"},
	}
}

func fakeNativeReadyGGUF(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	for _, value := range []any{uint32(3), uint64(11), uint64(9)} {
		if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeTestGGUFStringKV(t, &buf, "general.architecture", "llama")
	writeTestGGUFUint32KV(t, &buf, "general.file_type", 1)
	writeTestGGUFUint32KV(t, &buf, "llama.context_length", 128)
	writeTestGGUFUint32KV(t, &buf, "llama.embedding_length", 4)
	writeTestGGUFUint32KV(t, &buf, "llama.block_count", 1)
	writeTestGGUFUint32KV(t, &buf, "llama.feed_forward_length", 8)
	writeTestGGUFUint32KV(t, &buf, "llama.attention.head_count", 2)
	writeTestGGUFUint32KV(t, &buf, "llama.attention.head_count_kv", 1)
	writeTestGGUFStringArrayKV(t, &buf, "tokenizer.ggml.tokens", []string{"<unk>", "hello"})
	writeTestGGUFTensor(t, &buf, "token_embd.weight", []uint64{4, 2}, 1)
	writeTestGGUFTensor(t, &buf, "output_norm.weight", []uint64{4}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.attn_norm.weight", []uint64{4}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.attn_q.weight", []uint64{4, 4}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.attn_k.weight", []uint64{4, 2}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.attn_v.weight", []uint64{4, 2}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.attn_output.weight", []uint64{4, 4}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.ffn_norm.weight", []uint64{4}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.ffn_gate.weight", []uint64{4, 8}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.ffn_up.weight", []uint64{4, 8}, 1)
	writeTestGGUFTensor(t, &buf, "blk.0.ffn_down.weight", []uint64{8, 4}, 1)
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeTestGGUFStringKV(t *testing.T, buf *bytes.Buffer, key, value string) {
	t.Helper()
	writeTestGGUFString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(8)); err != nil {
		t.Fatal(err)
	}
	writeTestGGUFString(t, buf, value)
}

func writeTestGGUFUint32KV(t *testing.T, buf *bytes.Buffer, key string, value uint32) {
	t.Helper()
	writeTestGGUFString(t, buf, key)
	if err := binary.Write(buf, binary.LittleEndian, uint32(4)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
		t.Fatal(err)
	}
}

func writeTestGGUFStringArrayKV(t *testing.T, buf *bytes.Buffer, key string, values []string) {
	t.Helper()
	writeTestGGUFString(t, buf, key)
	for _, value := range []any{uint32(9), uint32(8), uint64(len(values))} {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range values {
		writeTestGGUFString(t, buf, value)
	}
}

func writeTestGGUFTensor(t *testing.T, buf *bytes.Buffer, name string, shape []uint64, typ uint32) {
	t.Helper()
	writeTestGGUFString(t, buf, name)
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(shape))); err != nil {
		t.Fatal(err)
	}
	for _, dim := range shape {
		if err := binary.Write(buf, binary.LittleEndian, dim); err != nil {
			t.Fatal(err)
		}
	}
	if err := binary.Write(buf, binary.LittleEndian, typ); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(buf, binary.LittleEndian, uint64(0)); err != nil {
		t.Fatal(err)
	}
}

func writeTestGGUFString(t *testing.T, buf *bytes.Buffer, value string) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	if _, err := buf.WriteString(value); err != nil {
		t.Fatal(err)
	}
}
