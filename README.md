# shenava-asr-server

[![Windows CPU-only](https://github.com/Reza2kn/shenava-asr-server/actions/workflows/windows-cpu.yml/badge.svg)](https://github.com/Reza2kn/shenava-asr-server/actions/workflows/windows-cpu.yml)

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

The first launch downloads the offline, streaming, and public Nemotron model assets from Hugging Face.
`HF_TOKEN` is optional. See
[the backend matrix](docs/BACKENDS.md) for build commands and deployment boundaries.

## Native Rust diarization and streaming

The optional native pipelines are implemented in Rust with Tract and selected per request:

- `native-diarization` runs the Nemotron-3 diarization graph and preserves speaking order,
  including overlapping speaker spans, when `/transcribe` receives `diarization=true`.
- `native-streaming` runs the cache-aware 114M Shenava Koochik CTC graph when
  `/transcribe` receives `mode=streaming`.

Build both features with CUDA support (the `gpu-or-cpu` backend falls back to CPU when CUDA is
unavailable):

```bash
cargo build --release --locked --features cuda,native-diarization,native-streaming
./target/release/shenava-asr-server \
  --backend gpu-or-cpu \
  --model models/model.onnx \
  --tokens models/tokens.txt \
  --mel assets/mel_filters.json \
  --diarizer-nemotron-model models/nemotron3-streaming.onnx \
  --diarizer-native-backend gpu-or-cpu \
  --streaming-model models/koochik-streaming.onnx \
  --streaming-tokens models/tokens.txt \
  --streaming-backend gpu-or-cpu
```

The verified 114M Koochik streaming package is
[Reza2kn/Shenava-Koochik-v1.0-tract-streaming](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-streaming).
The exported Nemotron package for the native runtime is
[Reza2kn/shenava-nemotron3-rust](https://huggingface.co/Reza2kn/shenava-nemotron3-rust)
(private). The running server requires only Rust and the selected model files; the one-time
NeMo export step is not part of the serving path. See [diarization details](docs/DIARIZATION.md),
[streaming details](docs/STREAMING.md), and the [Koochik runtime notes](docs/KOOCHIK_STREAMING_RUST.md).

### One-command launcher

`run.sh` now builds both native Rust features and stores their verified assets under `models/`.
The public derived Nemotron package requires no Hugging Face authentication:

```bash
./run.sh
```

The launcher downloads Nemotron anonymously and stores it under `models/`. `HF_TOKEN` may still be
set for a private mirror. To explicitly skip optional assets, use `SHENAVA_ENABLE_DIARIZATION=0` or
`SHENAVA_ENABLE_STREAMING=0`. The launcher always compiles `native-diarization` and
`native-streaming`, so a later request cannot fail merely because the binary was built CPU-only.

## Windows CPU-only: optimized build and launch

Use 64-bit Windows with the stable Rust MSVC toolchain, an MSVC linker, and the Windows SDK. The
recommended launcher downloads the pinned ONNX model and tokens, verifies their size and SHA-256,
builds the optimized release executable, and starts the server:

```powershell
git clone https://github.com/Reza2kn/shenava-asr-server.git
Set-Location shenava-asr-server
.\run-windows.ps1 -Addr "127.0.0.1:3000"
```

The resulting native binary is `target\release\shenava-asr-server.exe`. The running server does
not require Python, onnxruntime, CUDA, or a C++ model runtime.

To build the optimized executable manually after `models\model.onnx` and `models\tokens.txt` are
available:

```powershell
cargo build --release --locked --no-default-features --features cpu-only
```

Launch that executable directly:

```powershell
.\target\release\shenava-asr-server.exe `
  --backend cpu `
  --model .\models\model.onnx `
  --tokens .\models\tokens.txt `
  --mel .\assets\mel_filters.json `
  --addr 127.0.0.1:3000
```

Check that the model is loaded and the CPU backend is ready:

```powershell
Invoke-RestMethod http://127.0.0.1:3000/health
# ok backend version decoder_revision
# -- ------- ------- ----------------
# True cpu     0.1.1 sentencepiece-v2
```

To use startup hotwords, save one UTF-8 word or phrase per line and pass the file to the launcher:

```powershell
.\run-windows.ps1 -Addr "127.0.0.1:3000" -Hotwords ".\hotwords.txt"
```

Windows performance and deployment tips:

- Always use the `--release` binary. The release profile enables optimization level 3 and LTO;
  debug builds are much slower.
- Keep the server running between requests so model loading and tract graph analysis happen once.
- Bind to `127.0.0.1` for a local application. Use `0.0.0.0` only when other machines must connect,
  and configure Windows Firewall for the selected port.
- Keep each upload at about 20 seconds or less because the published model uses a fixed 2,005-frame
  input window.
- Every green [Windows CPU-only workflow run](https://github.com/Reza2kn/shenava-asr-server/actions/workflows/windows-cpu.yml)
  uploads a `shenava-windows-x86_64` artifact containing the optimized `.exe` and validation logs.
  The model remains a separate pinned download handled by `run-windows.ps1`.

## API

```text
POST /transcribe
  multipart `file` = WAV
  optional multipart `hotwords` = newline-delimited words/phrases
  optional multipart `diarization` = true/false (default false)
  optional multipart `mode` = offline/streaming (default offline)

  {
    "text": "hotbeam result, or greedy when no hotwords are present",
    "greedy": "plain CTC baseline",
    "elapsed_ms": 42,
    "backend": "cpu",
    "decoder": "hotbeam",
    "version": "0.1.1",
    "decoder_revision": "sentencepiece-v2",
    "mode": "offline",
    "diarization": false,
    "segments": null
  }

With `diarization=true` and the native Nemotron graph configured, `segments`
contains timestamped speaker spans and `text` is the same spans rendered as
ordered `speaker_N: text` lines. Overlapping spans stay overlapping and remain
adjacent in audio order. This composition uses the offline Koochik model for
each diarized span; `diarization=true` and `mode=streaming` are currently
rejected together.

With `mode=streaming`, configure `--streaming-model` and its matching
`--streaming-tokens`. The cache-aware Rust/Tract model can process audio longer
than the offline model's fixed 2,005-frame window.

When `mode` is omitted, a configured streaming model is selected automatically
for audio longer than the offline window. The response reports
`"mode":"streaming"` in that case. Set `mode=offline` to explicitly keep the
offline model's 10-second chunked path.

POST /diarize
  multipart `file` = WAV, optional multipart `model` = `sortformer` or `nemotron3`
  available with `--diarizer-worker`, or natively for Nemotron with
  `--features native-diarization --diarizer-nemotron-model <graph.onnx>`

  {
    "model": "sortformer",
    "backend": "cpu",
    "frame_ms": 80,
    "max_speakers": 4,
    "segments": [{"start_ms":1040,"end_ms":2240,"speaker_id":1,"speaker":"speaker_1"}],
    "elapsed_ms": 42
  }

GET /health
  {"ok":true,"backend":"cpu","version":"0.1.1","decoder_revision":"sentencepiece-v2"}
```

See [optional speaker diarization](docs/DIARIZATION.md) for the native-Rust
Nemotron graph contract and the ordered overlap-preserving segment stream.

WAV input may be mono or multichannel integer PCM (8–32 bit) or float32, at any non-zero sample
rate. The server resamples and mixes to 16 kHz mono. The published offline models have a fixed
2,005-frame window, so requests longer than about 20 seconds are rejected instead of silently
truncated.

```bash
curl -F file=@speech.wav \
  -F $'hotwords=شنوا\nرضا سیار' \
  http://127.0.0.1:3000/transcribe

# Native Rust Nemotron diarization + ordered speaker-attributed ASR
curl -F file=@speech.wav -F diarization=true \
  http://127.0.0.1:3000/transcribe

# Native Rust cache-aware Koochik streaming
curl -F file=@speech.wav -F mode=streaming \
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

### If Persian words are split into BPE pieces

Output such as `فرو ش نده سی ب` comes from Shenava server 0.1.0's old greedy renderer, which added
a space after every CTC token instead of only at SentencePiece `▁` word boundaries. It is not a
CPU accuracy difference. CPU, CUDA, and CoreML now use the same corrected decoder.

If `/transcribe` returns only `text`, `greedy`, and `elapsed_ms`, that process is stale. A current
response and `/health` both include `"version":"0.1.1"` and
`"decoder_revision":"sentencepiece-v2"`.

Update and rebuild the Ubuntu service, then restart the process that owns the listening port:

```bash
git pull --ff-only origin main
cargo build --release --locked
./target/release/shenava-asr-server --version
# shenava-asr-server 0.1.1
```

The one-command `./run.sh` path performs the same locked release rebuild. Do not keep an older
`target/release/shenava-asr-server` process running after pulling the fix.

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

1. `src/fbank.rs`: NeMo-compatible 16 kHz, 80-bin log-mel features; fixed offline `[1,80,2005]` or streaming chunks.
2. `src/model.rs`: offline tract ONNX or CoreML ML Program; returns `[T,1025]` CTC scores and valid length.
3. `src/streaming.rs`: cache-aware 114M Koochik CTC chunks for native Rust streaming.
4. `src/nemotron.rs`: native Rust Nemotron speaker-cache inference and ordered overlap-preserving spans.
5. `src/decode.rs`: correct SentencePiece/CTC greedy baseline and optional hotword CTC beam search.

The tract model is
[`Reza2kn/Shenava-Koochik-v1.0-tract-offline`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline).
The Apple model is
[`Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16).
Both use ve_tok_v4 with blank id 1024 and the same fixed feature window.

## License

Apache-2.0
