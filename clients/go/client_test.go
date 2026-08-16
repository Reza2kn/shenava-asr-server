package shenava

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestTranscribeSendsAudioAndHotwords(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		audio, _ := io.ReadAll(file)
		if string(audio) != "RIFF-test" {
			t.Fatalf("unexpected audio: %q", audio)
		}
		if got := r.FormValue("hotwords"); got != "شنوا\nرضا" {
			t.Fatalf("unexpected hotwords: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"شنوا","greedy":"شنوا","elapsed_ms":9,"backend":"cpu","decoder":"hotbeam","version":"0.1.1","decoder_revision":"sentencepiece-v2"}`)
	}))
	defer server.Close()

	client := New(server.URL)
	got, err := client.Transcribe(
		context.Background(),
		"speech.wav",
		strings.NewReader("RIFF-test"),
		TranscribeOptions{Hotwords: []string{"شنوا", "رضا"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got.Decoder != "hotbeam" || got.Backend != "cpu" ||
		got.Version != "0.1.1" || got.DecoderRevision != "sentencepiece-v2" {
		t.Fatalf("unexpected response: %#v", got)
	}
}
