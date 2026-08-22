package server

import (
	"encoding/json"
	"fmt"
	"net/http"

	nego "github.com/gakon/nego-ai"
)

type HandlerOptions struct {
	ModelID string
	Model   nego.Model
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
		handleCompletion(w, r, modelID, opts.Model)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			methodNotAllowed(w)
			return
		}
		handleChatCompletion(w, r, modelID, opts.Model)
	})
	return mux, nil
}

type completionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	Stop        []string `json:"stop"`
	Stream      bool     `json:"stream"`
}

type chatRequest struct {
	Model       string         `json:"model"`
	Messages    []nego.Message `json:"messages"`
	MaxTokens   int            `json:"max_tokens"`
	Temperature float64        `json:"temperature"`
	TopP        float64        `json:"top_p"`
	Stop        []string       `json:"stop"`
	Stream      bool           `json:"stream"`
}

func handleCompletion(w http.ResponseWriter, r *http.Request, modelID string, model nego.Model) {
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	out, err := model.Generate(r.Context(), nego.GenerateRequest{
		Prompt:      req.Prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
	})
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

func handleChatCompletion(w http.ResponseWriter, r *http.Request, modelID string, model nego.Model) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	chatReq := nego.ChatRequest{
		Messages:    req.Messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
	}
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
