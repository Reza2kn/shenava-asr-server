# Shenava Go client

This package lets a Go service use the Shenava Rust server as a local or remote sidecar. It uses
only the Go standard library and does not require CGo.

```go
package main

import (
    "context"
    "fmt"
    "os"

    shenava "github.com/Reza2kn/shenava-asr-server/clients/go"
)

func main() {
    wav, err := os.Open("speech.wav")
    if err != nil { panic(err) }
    defer wav.Close()

    client := shenava.New("http://127.0.0.1:3000")
    result, err := client.Transcribe(context.Background(), "speech.wav", wav,
        shenava.TranscribeOptions{Hotwords: []string{"شنوا", "رضا سیار"}})
    if err != nil { panic(err) }
    fmt.Println(result.Text, result.Backend, result.Decoder, result.DecoderRevision)
}
```

`Version` and `DecoderRevision` are also returned by `Health`. Require
`DecoderRevision == "sentencepiece-v2"` when a deployment must reject the old renderer that spaced
every Persian BPE piece.

Run `go test ./...` from this directory.
