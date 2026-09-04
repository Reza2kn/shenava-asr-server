# Native Rust Koochik streaming artifact

This package is the fixed-shape ONNX CTC streaming branch used by
`shenava-asr-server` with `mode=streaming`. It is the 114M Persian Koochik
cache-aware graph and runs in Rust through Tract. NeMo/Python
is only needed for the one-time export; the server runtime uses no Python,
C++, or onnxruntime.

Contract:

- 16 kHz mono input with the server's 80-bin NeMo fbank
- feature chunks of 121 frames, shifted by 112 frames (about 1.12 seconds),
  with 9 frames of pre-encode overlap
- encoder cache `[1, 17, 70, 512]`
- time cache `[1, 17, 512, 8]`
- 1,025 CTC classes, blank id 1,024
- matching Koochik `tokens.txt` tokenizer

The model can decode audio longer than the offline model's fixed 2,005-frame
window because state is carried across chunks. Responses still expose the
same `text`, `greedy`, `elapsed_ms`, and decoder fields, with
`"mode":"streaming"` and `"diarization":false`.

Use it with:

```bash
cargo build --release --features cuda,native-streaming
./shenava-asr-server \
  --model models/model.onnx \
  --streaming-model models/koochik-streaming-fp32.onnx \
  --streaming-tokens models/tokens.txt \
  --streaming-backend gpu-or-cpu
```

The original checkpoint and tokenizer remain in the source model repository;
this graph is the Rust deployment derivative.
