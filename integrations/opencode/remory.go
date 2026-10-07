// Package remory provides the provider/Summarize seam for opencode-ai/opencode.
package remory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	URL, APIKey, SessionID string
	HTTP                   *http.Client
}

type CompactRequest struct {
	PrefixIDs  []int   `json:"prefix_ids"`
	HistoryIDs []int   `json:"history_ids"`
	SummaryIDs []int   `json:"summary_ids"`
	Previous   *string `json:"previous,omitempty"`
}

type Checkpoint struct {
	Handle     string         `json:"handle"`
	SummaryIDs []int          `json:"summary_ids"`
	Receipt    map[string]any `json:"receipt"`
}

type GenerateRequest struct {
	Handle          string         `json:"handle,omitempty"`
	InputIDs        []int          `json:"input_ids,omitempty"`
	ContinuationIDs []int          `json:"continuation_ids,omitempty"`
	MaxNewTokens    int            `json:"max_new_tokens"`
	Sampling        map[string]any `json:"sampling,omitempty"`
}

type Generation struct {
	Text         string         `json:"text"`
	OutputIDs    []int          `json:"output_ids"`
	Usage        map[string]any `json:"usage"`
	FinishReason any            `json:"finish_reason"`
}

func (c *Client) post(ctx context.Context, route string, input, output any) error {
	body, err := json.Marshal(input)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(c.URL, "/")+route, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("X-Remory-Session", c.SessionID)
	req.Header.Set("Content-Type", "application/json")
	client := c.HTTP
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("Remory %d: %s", resp.StatusCode, body)
	}
	return json.NewDecoder(resp.Body).Decode(output)
}

// Compact must finish before Summarize creates/replaces the continuation session.
// Persist Handle with that session; a failed call must leave its source intact.
func (c *Client) Compact(ctx context.Context, request CompactRequest) (Checkpoint, error) {
	var output Checkpoint
	err := c.post(ctx, "/v1/compact", request, &output)
	return output, err
}

func (c *Client) Generate(ctx context.Context, request GenerateRequest) (Generation, error) {
	var output Generation
	err := c.post(ctx, "/v1/generate", request, &output)
	return output, err
}
