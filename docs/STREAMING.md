# Native Rust streaming Koochik

The verified 114M model package is
[Reza2kn/Shenava-Koochik-v1.0-tract-streaming](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-streaming).
Its FP32 `model.onnx` is the reference graph for this route; the repository's
INT4 graph currently advertises a 121-frame contract but contains a 25-frame
input, so it remains excluded until that package is corrected.

The server supports a second ASR mode through `POST /transcribe`:

```bash
cargo build --release --features cuda,native-streaming
./target/release/shenava-asr-server \
  --backend gpu-or-cpu \
  --model models/model.onnx \
  --streaming-model models/koochik-streaming-fp32.onnx \
  --streaming-tokens models/tokens.txt \
  --streaming-backend gpu-or-cpu
```

Requests that omit `mode` use the existing fixed-window offline Shenava
model for short audio. For long audio, when the streaming graph is configured,
the server automatically selects streaming if `mode` is omitted. Requests with
`mode=streaming` use the cache-aware Persian 114M Koochik
CTC graph, carrying the encoder cache across 121-frame chunks shifted by 112
frames (about 1.12 seconds, with 9 frames of pre-encode overlap). This
supports audio longer than the offline 2,005-frame window.

Use `mode=offline` to explicitly force the offline model's 10-second chunked
path, for compatibility comparisons or deployments without the streaming graph.

The streaming graph and tokenizer are a matched pair. The graph uses ONNX
inputs `[1,80,121]`, `[1]`, `[1,17,70,512]`, `[1,17,512,8]`, and `[1]`; its
CTC output has 1,025 classes with blank id 1,024. Use the matching Koochik
`tokens.txt` from the streaming model package.

The runtime is native Rust and Tract. NeMo/Python is only an export-time
dependency used to produce the ONNX graph.
