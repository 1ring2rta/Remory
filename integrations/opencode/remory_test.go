package remory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompactAndGenerate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Remory-Session") != "stable-owner" || r.Header.Get("Authorization") != "Bearer key" {
			t.Error("missing owner or auth")
		}
		if r.URL.Path == "/v1/compact" {
			var body CompactRequest
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if len(body.HistoryIDs) != 2 || body.PrefixIDs[0] != 1 {
				t.Error("source changed")
			}
			json.NewEncoder(w).Encode(Checkpoint{Handle: "test-handle", SummaryIDs: body.SummaryIDs})
		} else {
			var body GenerateRequest
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body.Handle != "test-handle" || body.ContinuationIDs[0] != 9 {
				t.Error("checkpoint lost")
			}
			json.NewEncoder(w).Encode(Generation{Text: "ok", OutputIDs: []int{2}})
		}
	}))
	defer srv.Close()
	c := Client{URL: srv.URL, APIKey: "key", SessionID: "stable-owner"}
	checkpoint, err := c.Compact(context.Background(), CompactRequest{PrefixIDs: []int{1}, HistoryIDs: []int{2, 3}, SummaryIDs: []int{4}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := c.Generate(context.Background(), GenerateRequest{Handle: checkpoint.Handle, ContinuationIDs: []int{9}, MaxNewTokens: 10})
	if err != nil || result.Text != "ok" {
		t.Fatalf("%v %v", result, err)
	}
}

func TestErrorsAreReturnedBeforeCallerCommits(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "failure", 502) }))
	defer srv.Close()
	c := Client{URL: srv.URL}
	if _, err := c.Compact(context.Background(), CompactRequest{}); err == nil {
		t.Fatal("backend failure ignored")
	}
}
