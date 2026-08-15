# shenava-asr-server

Portable Shenava Koochik ASR behind one HTTP contract. Audio preprocessing and CTC decoding stay in
Rust; the acoustic-model runtime can be pure-Rust tract or native Apple CoreML.

No Python, onnxruntime, or C++ is used by the running server.

## Quick start

```bash
# Linux CPU, NVIDIA CUDA auto-detection, or an existing Unix deployment
./run.sh

# Windows CPU-only (PowerShell; downloads pinned, SHA-verified assets)
.\run-windows.ps1

# macOS 13+ native CoreML (CPU/GPU/Apple Neural Engine)
./run-apple.sh
```

The first launch downloads the selected model from Hugging Face. See
[the backend matrix](docs/BACKENDS.md) for build commands and deployment boundaries.

## API

```text
POST /transcribe
  multipart `file` = WAV
  optional multipart `hotwords` = newline-delimited words/phrases

  {
    "text": "hotbeam result, or greedy when no hotwords are present",
    "greedy": "plain CTC baseline",
    "elapsed_ms": 42,
    "backend": "cpu",
    "decoder": "hotbeam"
  }

GET /health
  {"ok":true,"backend":"cpu"}
```

WAV input may be mono or multichannel integer PCM (8–32 bit) or float32, at any non-zero sample
rate. The server resamples and mixes to 16 kHz mono. The published offline models have a fixed
2,005-frame window, so requests longer than about 20 seconds are rejected instead of silently
truncated.

```bash
curl -F file=@speech.wav \
  -F $'hotwords=شنوا\nرضا سیار' \
  http://127.0.0.1:3000/transcribe
```

Startup hotwords are also supported:

```bash
./run.sh --hotwords hotwords.txt --hotword-weight 2.5 --beam 80
```

Read [Improving word accuracy](docs/DECODING.md) or the
[Persian guide to improving word accuracy](docs/DECODING.fa.md) before tuning a hotword weight or
connecting a language model. `shenava-ctc-beam` is deliberately a no-LM hotbeam decoder; the
guides mark the custom-LM integration boundary explicitly.

## Go services

The dependency-free [Go client](clients/go) runs Shenava as a Rust sidecar and preserves the same
backend and model behavior:

```go
client := shenava.New("http://127.0.0.1:3000")
result, err := client.Transcribe(ctx, "speech.wav", wav, shenava.TranscribeOptions{
    Hotwords: []string{"شنوا", "رضا سیار"},
})
```

This avoids reimplementing the 114M FastConformer and fbank in Go while still fitting a normal Go
service deployment. See [clients/go/README.md](clients/go/README.md).

## Shared inference contract

1. `src/fbank.rs`: NeMo-compatible 16 kHz, 80-bin log-mel features; fixed `[1,80,2005]` tensor.
2. `src/model.rs`: tract ONNX or CoreML ML Program; returns `[T,1025]` CTC scores and valid length.
3. `src/decode.rs`: correct SentencePiece/CTC greedy baseline and optional hotword CTC beam search.

The tract model is
[`Reza2kn/Shenava-Koochik-v1.0-tract-offline`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline).
The Apple model is
[`Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16).
Both use ve_tok_v4 with blank id 1024 and the same fixed feature window.

## License

Apache-2.0
