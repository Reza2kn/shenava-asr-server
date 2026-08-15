# Backends

All backends feed the same Rust fbank and decoder. A backend changes inference placement, not the
HTTP response or tokenization.

| Target | CLI backend | Model | Runtime boundary |
|---|---|---|---|
| Windows CPU-only | `cpu` | fixed Koochik ONNX | pure Rust tract; no DLL model runtime |
| Linux CPU | `cpu` | fixed Koochik ONNX | pure Rust tract |
| NVIDIA Linux | `gpu-or-cpu` | fixed Koochik ONNX | tract CUDA feature and CUDA driver/toolkit |
| macOS 13+ | `coreml` | fixed Koochik `.mlpackage`/`.mlmodelc` | Rust bindings to the system CoreML framework |
| Generic Go service | whichever Rust sidecar target fits the host | same as sidecar | dependency-free Go HTTP client |

## Windows CPU-only

Install stable Rust with the MSVC toolchain, then run:

```powershell
.\run-windows.ps1 -Addr "127.0.0.1:3000" -Hotwords ".\hotwords.txt"
```

The script pins the Hugging Face revision and verifies the ONNX and token files by size and
SHA-256 before building with the dedicated `cpu-only` feature. The produced server uses tract's
CPU runtime. It does not load onnxruntime, Python, CUDA, or a C++ inference library.

Every pull request and push to `main` runs `.github/workflows/windows-cpu.yml` on GitHub's native
Windows/MSVC runner. It tests the CPU-only feature, builds and launches the release `.exe`, starts
the server through `run-windows.ps1`, and requires a healthy CPU-backend response.

Manual build:

```powershell
cargo build --release --locked --no-default-features --features cpu-only
.\target\release\shenava-asr-server.exe `
  --backend cpu --model models\model.onnx --tokens models\tokens.txt
```

## Apple CoreML

`run-apple.sh` downloads the published fixed-window FP16 CoreML package at a pinned revision,
verifies every package payload, enables the `coreml` Cargo feature, and selects all CoreML compute
units (CPU/GPU/ANE):

```bash
./run-apple.sh --addr 127.0.0.1:3000
```

The Rust process can load a compiled `.mlmodelc` directly. When given `.mlmodel` or `.mlpackage`,
it asks the system CoreML framework to compile it at startup:

```bash
cargo build --release --locked --no-default-features --features coreml
./target/release/shenava-asr-server \
  --backend coreml \
  --model /path/to/shenava-koochik-v1.0_ctc_fixed2005_len_fp16.mlmodelc \
  --tokens models/tokens.txt
```

The checked contract is:

- `processed_signal`: float32 `[1,80,2005]`
- `processed_signal_length`: int32 `[1]`
- `logits`: float16 `[1,252,1025]` (converted to float32 for decoding)
- `encoded_lengths`: int32 `[1]`

This repository currently ships a macOS HTTP service. The same CoreML model contract is suitable
for an iOS/iPadOS app, but axum server packaging is not an iOS application integration.

## NVIDIA

```bash
cargo build --release --locked --features cuda
./target/release/shenava-asr-server --backend gpu-or-cpu ...
```

See `deploy/README.md` for the current CUDA/NVRTC host requirements.

## Go deployment

Run the Rust server in the same pod, VM, Windows service, or host process group as the Go service.
Point `clients/go` at its loopback address. The Go client supports health checks, WAV upload, and
per-request hotwords without CGo or platform-specific model bindings.
