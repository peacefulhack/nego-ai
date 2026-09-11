package server

import (
	"encoding/json"
	"fmt"
	"net/http"

	nego "github.com/gakon/nego-ai"
	"github.com/gakon/nego-ai/modelinfo"
)

type HandlerOptions struct {
	ModelID    string
	Model      nego.Model
	Generation *modelinfo.GenerationConfig
}

func NewHandler(opts HandlerOptions) (http.Handler, error) {
	if opts.Model == nil {
		return nil, fmt.Errorf("model is required")
	}
	modelID := opts.ModelID
	if modelID == "" {
		modelID = "nego-model"
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		writeJSON(w, map[string]any{
			"object": "list",
			"data": []map[string]any{{
				"id":       modelID,
				"object":   "model",
				"owned_by": "nego",
			}},
		})
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			methodNotAllowed(w)
			return
		}
		handleCompletion(w, r, modelID, opts.Model, opts.Generation)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			methodNotAllowed(w)
			return
		}
		handleChatCompletion(w, r, modelID, opts.Model, opts.Generation)
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			methodNotAllowed(w)
			return
		}
		handleEmbeddings(w, r, modelID, opts.Model)
	})
	return mux, nil
}

type completionRequest struct {
	Model         string   `json:"model"`
	Prompt        string   `json:"prompt"`
	MaxTokens     *int     `json:"max_tokens"`
	Temperature   *float64 `json:"temperature"`
	TopP          *float64 `json:"top_p"`
	RepeatPenalty *float64 `json:"repeat_penalty"`
	Stop          []string `json:"stop"`
	Seed          int64    `json:"seed"`
	Stream        bool     `json:"stream"`
}

type chatRequest struct {
	Model         string         `json:"model"`
	Messages      []nego.Message `json:"messages"`
	MaxTokens     *int           `json:"max_tokens"`
	Temperature   *float64       `json:"temperature"`
	TopP          *float64       `json:"top_p"`
	RepeatPenalty *float64       `json:"repeat_penalty"`
	Stop          []string       `json:"stop"`
	Seed          int64          `json:"seed"`
	Stream        bool           `json:"stream"`
}

type embeddingsRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

func handleCompletion(w http.ResponseWriter, r *http.Request, modelID string, model nego.Model, generation *modelinfo.GenerationConfig) {
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	genReq := nego.GenerateRequest{
		Prompt: req.Prompt,
		Seed:   req.Seed,
	}
	applyCompletionRequestOptions(&genReq, req, generation)
	if req.Stream {
		stream, err := nego.StreamGenerate(r.Context(), model, genReq)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeCompletionStream(w, modelID, stream)
		return
	}
	out, err := model.Generate(r.Context(), genReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, map[string]any{
		"id":     "cmpl-nego",
		"object": "text_completion",
		"model":  modelID,
		"choices": []map[string]any{{
			"text":          out.Text,
			"index":         0,
			"finish_reason": "stop",
		}},
	})
}

func applyCompletionRequestOptions(out *nego.GenerateRequest, req completionRequest, generation *modelinfo.GenerationConfig) {
	if req.MaxTokens != nil {
		out.MaxTokens = *req.MaxTokens
	} else if generation != nil && generation.MaxNewTokens != nil {
		out.MaxTokens = *generation.MaxNewTokens
	}
	if req.Temperature != nil {
		out.Temperature = *req.Temperature
	} else if generation != nil && generation.Temperature != nil {
		out.Temperature = *generation.Temperature
	}
	if req.TopP != nil {
		out.TopP = *req.TopP
	} else if generation != nil && generation.TopP != nil {
		out.TopP = *generation.TopP
	}
	if req.RepeatPenalty != nil {
		out.RepeatPenalty = *req.RepeatPenalty
	} else if generation != nil && generation.RepetitionPenalty != nil {
		out.RepeatPenalty = *generation.RepetitionPenalty
	}
	if req.Stop != nil {
		out.Stop = append([]string(nil), req.Stop...)
	} else if generation != nil && len(generation.StopStrings) > 0 {
		out.Stop = append([]string(nil), generation.StopStrings...)
	}
}

func writeCompletionStream(w http.ResponseWriter, modelID string, stream nego.Stream) {
	defer stream.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	for token := range stream.Tokens() {
		chunk := map[string]any{
			"id":     "cmpl-nego",
			"object": "text_completion",
			"model":  modelID,
			"choices": []map[string]any{{
				"text":  token.Text,
				"index": 0,
			}},
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			break
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func handleChatCompletion(w http.ResponseWriter, r *http.Request, modelID string, model nego.Model, generation *modelinfo.GenerationConfig) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	chatReq := nego.ChatRequest{
		Messages: req.Messages,
		Seed:     req.Seed,
	}
	applyChatRequestOptions(&chatReq, req, generation)
	if req.Stream {
		stream, err := model.StreamChat(r.Context(), chatReq)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeChatStream(w, modelID, stream)
		return
	}
	resp, err := model.Chat(r.Context(), chatReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, map[string]any{
		"id":     "chatcmpl-nego",
		"object": "chat.completion",
		"model":  modelID,
		"choices": []map[string]any{{
			"index":         0,
			"message":       resp.Message,
			"finish_reason": "stop",
		}},
	})
}

func applyChatRequestOptions(out *nego.ChatRequest, req chatRequest, generation *modelinfo.GenerationConfig) {
	if req.MaxTokens != nil {
		out.MaxTokens = *req.MaxTokens
	} else if generation != nil && generation.MaxNewTokens != nil {
		out.MaxTokens = *generation.MaxNewTokens
	}
	if req.Temperature != nil {
		out.Temperature = *req.Temperature
	} else if generation != nil && generation.Temperature != nil {
		out.Temperature = *generation.Temperature
	}
	if req.TopP != nil {
		out.TopP = *req.TopP
	} else if generation != nil && generation.TopP != nil {
		out.TopP = *generation.TopP
	}
	if req.RepeatPenalty != nil {
		out.RepeatPenalty = *req.RepeatPenalty
	} else if generation != nil && generation.RepetitionPenalty != nil {
		out.RepeatPenalty = *generation.RepetitionPenalty
	}
	if req.Stop != nil {
		out.Stop = append([]string(nil), req.Stop...)
	} else if generation != nil && len(generation.StopStrings) > 0 {
		out.Stop = append([]string(nil), generation.StopStrings...)
	}
}

func writeChatStream(w http.ResponseWriter, modelID string, stream nego.Stream) {
	defer stream.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	for token := range stream.Tokens() {
		chunk := map[string]any{
			"id":     "chatcmpl-nego",
			"object": "chat.completion.chunk",
			"model":  modelID,
			"choices": []map[string]any{{
				"index": 0,
				"delta": map[string]string{
					"content": token.Text,
				},
			}},
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			break
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request, modelID string, model nego.Model) {
	var req embeddingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	resp, err := nego.Embed(r.Context(), model, nego.EmbeddingRequest{Input: req.Input})
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	data := make([]map[string]any, 0, len(resp.Embeddings))
	for i, embedding := range resp.Embeddings {
		data = append(data, map[string]any{
			"object":    "embedding",
			"index":     i,
			"embedding": embedding,
		})
	}
	writeJSON(w, map[string]any{
		"object": "list",
		"model":  modelID,
		"data":   data,
	})
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"message": message,
		},
	})
}

func methodNotAllowed(w http.ResponseWriter) {
	writeError(w, http.StatusMethodNotAllowed, "method not allowed")
}
