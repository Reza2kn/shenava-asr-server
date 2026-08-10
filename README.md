# shenava-asr-server

Fully-Rust Shenava ASR HTTP server. Loads the Shenava **Koochik** (114M FastConformer CTC) model
through [tract](https://github.com/sonos/tract) (pure-Rust ONNX inference), computes the NeMo
log-mel fbank in Rust, and decodes with hotword-boosted CTC beam search via
[`shenava-ctc-beam`](https://github.com/Reza2kn/shenava-ctc-beam). Served over HTTP with axum.

No Python, no onnxruntime, no C++ — one `./run.sh`.

## Quick start

```bash
./run.sh                 # downloads model + tokens, builds, starts on :3000
./run.sh --addr 0.0.0.0:8080 --hotwords hotwords.txt
```

On first run it downloads the pre-simplified model (~418 MB) from
`Reza2kn/Shenava-Koochik-v1.0-tract-offline`. The model was run through `onnxsim` to constant-fold
the NeMo dynamic-shape ops so tract can analyse it. Requires only Rust (cargo).

## Backends

`--backend gpu-or-cpu` (default) uses tract's `gpu-or-cpu` runtime: Metal on Apple silicon, CUDA on
NVIDIA (build with `--features cuda`), else CPU. `--backend cpu` forces CPU.

```bash
# CUDA (NVIDIA hosts; runtime-detected, no nvcc needed):
cargo build --release --features cuda
./target/release/shenava-asr-server --backend gpu-or-cpu ...

# CPU anywhere:
cargo build --release
./target/release/shenava-asr-server --backend cpu ...
```

`./run.sh` auto-detects `nvidia-smi` and enables the CUDA feature + `gpu-or-cpu` backend.

## API

```
POST /transcribe   multipart form field `file` = 16 kHz mono WAV (any sample rate OK)
                   -> {"text": "...", "greedy": "...", "elapsed_ms": 42}
GET  /health       -> {"ok": true}
```

- `text` — hotword-boosted beam decode (if `--hotwords` given), else greedy.
- `greedy` — plain CTC greedy decode.

## How it works

1. **fbank** (`src/fbank.rs`): NeMo `AudioToMelSpectrogramPreprocessor` reproduction —
   16 kHz, n_fft 512, hop 160, hann(periodic=false), center pad 256 reflect, preemphasis 0.97,
   Slaney 80×257 mel, power spectrum, natural log, `normalize=NA`. Matches the deployed
   reference (`koochik_server.py` + `preprocessor.json`) to the sample.
2. **tract** (`src/model.rs`): loads the pre-simplified ONNX (fixed `[1,80,2005]` input),
   runs on the chosen backend → `log_probs [1,T',1025]` + `output_length`. Numerically identical
   to onnxruntime (max abs diff ~3e-5, 100% argmax agreement).
3. **decode** (`src/decode.rs`): `shenava-ctc-beam` hotword-boosted CTC prefix beam search
   (beam 80, weight 2.5); BPE `▁` word-boundary handling; `<...>` special tokens remapped to PUA
   and stripped.

## Model

`Reza2kn/Shenava-Koochik-v1.0-tract-offline` — the sherpa-onnx Koochik export, pre-simplified
with `onnxsim` at fixed `[1,80,2005]`. Requires the `shenava` branch of `Reza2kn/tract`
(relaxes tract's i64/TDim/shape inference for NeMo FastConformer exports).

## License

Apache-2.0
