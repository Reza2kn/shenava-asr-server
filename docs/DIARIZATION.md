# Optional speaker diarization

The Rust server exposes an optional `POST /diarize` route and an integrated
`diarization=true` option on `POST /transcribe`. The preferred Nemotron path is
a native Rust Tract runtime; the worker boundary remains available for other
experiments.

## Native Rust Nemotron

Build with `--features native-diarization` and pass a fixed-shape ONNX export
of `nvidia/Nemotron-3-Diarization-preview`:

```bash
cargo build --release --features cuda,native-diarization
./target/release/shenava-asr-server \
  --backend gpu-or-cpu \
  --diarizer-nemotron-model models/nemotron3-streaming.onnx \
  --diarizer-native-backend gpu-or-cpu
```

The exported graph consumes 128-bin NeMo mel features plus speaker-cache and
FIFO state. Rust owns feature extraction, cache compression, model execution,
and output ordering. The graph is loaded once at startup; the `.silence.bin`
sidecar next to it contains the learned silence embedding used when the cache
compresses. CPU uses Tract’s optimized CPU kernels; NVIDIA hosts can select
Tract’s CUDA runtime.

The result is a flat event stream sorted by `(start_ms, end_ms, speaker_id)`.
That keeps speaking order while retaining simultaneous speakers as overlapping
segments rather than regrouping the result by speaker.

The worker receives:

```text
--model <sortformer|nemotron3>
--audio <temporary WAV path>
--backend <auto|cpu|cuda>
--preset <very_high_latency|high_latency|low_latency>
```

It prints one JSON object to stdout:

```json
{
  "model": "sortformer",
  "backend": "cpu",
  "frame_ms": 80,
  "max_speakers": 4,
  "segments": [
    {"start_ms": 1040, "end_ms": 2240, "speaker_id": 1, "speaker": "speaker_1"}
  ]
}
```

Example worker launch (legacy experiment boundary):

```bash
./target/release/shenava-asr-server \
  --backend gpu-or-cpu \
  --diarizer-worker ./scripts/diarize_worker.py \
  --diarizer-backend auto
```

Then:

```bash
curl -F file=@speech.wav http://127.0.0.1:3000/diarize
curl -F model=nemotron3 -F file=@speech.wav http://127.0.0.1:3000/diarize
```

The integrated transcription request runs Nemotron once, slices the original
WAV once per ordered span, transcribes each slice with offline Koochik, and
returns both `text` (one `speaker_N: ...` line per non-empty result) and
`segments` (the authoritative timestamped result). Segments are sorted by
`(start_ms, end_ms, speaker_id)`, so overlapping speakers are retained in the
audio timeline instead of being grouped by identity. Speaker labels are
arrival-order labels rather than cross-recording identities.

```bash
curl -F diarization=true -F file=@speech.wav \
  http://127.0.0.1:3000/transcribe
```

The streaming Koochik path is selected separately with
`-F mode=streaming`; combining it with diarization is rejected until a
streaming segment-transcription policy is added.
