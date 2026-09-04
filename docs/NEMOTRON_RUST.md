# Native Rust Nemotron diarization artifact

This package is a fixed-shape ONNX deployment artifact for the native Rust
`shenava-asr-server` diarization runtime. It was exported from the gated
`nvidia/Nemotron-3-Diarization-preview` checkpoint and is intended to be run
with Tract; the server does not load Python, C++, or onnxruntime at inference
time.

The graph contract is:

- 16 kHz mono audio, extracted into 128-bin NeMo-compatible log-mel frames
- 340 core frames plus 40 right-context frames, stacked by 8
- speaker cache `[1, 264, 512]` and FIFO `[1, 40, 512]`
- eight speaker channels
- state predictions, 10 ms high-resolution predictions, chunk embeddings,
  and output length

The `.silence.bin` file is a learned silence embedding used by the Rust cache
compression path and must remain next to the ONNX graph. The export boundary
used NeMo once; the runtime path is pure Rust.

The output is converted into contiguous per-speaker spans and globally sorted
by `(start_ms, end_ms, speaker_id)`. Overlapping spans remain overlapping, so
the transcript can preserve the order in which speakers appear in the audio.

Source model: [`nvidia/Nemotron-3-Diarization-preview`](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview)

Use with:

```bash
cargo build --release --features cuda,native-diarization
./shenava-asr-server \
  --diarizer-nemotron-model nemotron3-streaming.onnx \
  --diarizer-native-backend gpu-or-cpu
```

The source checkpoint's NVIDIA model terms and access restrictions continue
to apply to this derived deployment artifact.
