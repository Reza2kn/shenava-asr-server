// Package shenava is a dependency-free Go client for the Shenava Rust server.
// The acoustic model still runs in the Rust sidecar; Go services keep their
// normal deployment model and call the stable HTTP API.
package shenava

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

type TranscribeOptions struct {
	// Hotwords are applied only to this request and are combined with the
	// server's startup hotword file.
	Hotwords []string
}

type Transcription struct {
	Text      string `json:"text"`
	Greedy    string `json:"greedy"`
	ElapsedMS uint64 `json:"elapsed_ms"`
	Backend   string `json:"backend"`
	Decoder   string `json:"decoder"`
}

type Health struct {
	OK      bool   `json:"ok"`
	Backend string `json:"backend"`
}

func New(baseURL string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		HTTPClient: &http.Client{
			Timeout: 2 * time.Minute,
		},
	}
}

func (c *Client) Transcribe(
	ctx context.Context,
	filename string,
	wav io.Reader,
	options TranscribeOptions,
) (Transcription, error) {
	var result Transcription
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, err := form.CreateFormFile("file", filename)
	if err != nil {
		return result, fmt.Errorf("create WAV form field: %w", err)
	}
	if _, err := io.Copy(part, wav); err != nil {
		return result, fmt.Errorf("copy WAV: %w", err)
	}
	if len(options.Hotwords) > 0 {
		if err := form.WriteField("hotwords", strings.Join(options.Hotwords, "\n")); err != nil {
			return result, fmt.Errorf("write hotwords: %w", err)
		}
	}
	if err := form.Close(); err != nil {
		return result, fmt.Errorf("finish multipart request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/transcribe", &body)
	if err != nil {
		return result, err
	}
	req.Header.Set("Content-Type", form.FormDataContentType())
	if err := c.doJSON(req, &result); err != nil {
		return result, err
	}
	return result, nil
}

func (c *Client) Health(ctx context.Context) (Health, error) {
	var result Health
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/health", nil)
	if err != nil {
		return result, err
	}
	if err := c.doJSON(req, &result); err != nil {
		return result, err
	}
	return result, nil
}

func (c *Client) doJSON(req *http.Request, target any) error {
	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return fmt.Errorf("shenava returned %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}
	if err := json.NewDecoder(resp.Body).Decode(target); err != nil {
		return fmt.Errorf("decode Shenava response: %w", err)
	}
	return nil
}
