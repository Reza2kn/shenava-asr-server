# Live microphone input

Start the server using `./run.sh` on Linux, `./run-apple.sh` on macOS, or
`.\run-windows.ps1` on Windows, then open **http://localhost:3000**.
Choose a microphone, click **Start microphone**, grant browser permission, and
speak. Click **Stop** to release the microphone and flush the final words.
The transcript can be copied with **Copy transcript**.

The browser needs HTTPS when accessing a server on another machine. Localhost
works over HTTP. For a remote server, an SSH tunnel also works:

```sh
ssh -N -L 3000:127.0.0.1:3000 user@server
# Open http://localhost:3000 on your own computer.
```

There are no browser packages, Python dependencies, or separate frontend build.
The HTML, JavaScript, and audio worklet are embedded in the Rust executable.
Audio is not written to disk by this client or endpoint.

## Runtime

Live input requires `native-streaming` and the verified FP32 streaming graph.
All three launchers download the checksum-verified graph and enable this feature.
Mac uses CoreML for file transcription and the Tract CPU streaming graph for live
microphone input. Windows uses Tract CPU. Linux's launcher selects CUDA when
available. CPU performance depends on the machine; use an optimized release build.
To build only CPU ASR and microphone support manually:

```sh
cargo build --release --locked --no-default-features --features cpu-only,native-streaming
./target/release/shenava-asr-server \
  --backend cpu --model models/model.onnx --tokens models/tokens.txt \
  --streaming-model models/koochik-streaming.onnx \
  --streaming-tokens models/tokens.txt --streaming-backend cpu
```

The page reports a missing streaming model before requesting microphone access.
Existing manually built offline-only binaries must be rebuilt and configured.

## Efficient audio path

The browser's AudioContext converts the microphone's native sample rate to 16 kHz.
An AudioWorklet mixes channels and sends 200 ms packets of little-endian PCM16
(6,400 bytes; 32 KB/s). This avoids WAV uploads, base64 and repeated transcription
of earlier audio. There is no sound playback or microphone feedback.

Each connection owns its encoder cache and CTC decoder state, while model weights
are shared. A bounded PCM window retains the centered FFT and preemphasis context.
Each 121-frame model chunk advances 112 frames, about 1.12 seconds. The first chunk
needs about 1.216 seconds of audio, plus packet delivery and inference time. Stop
flushes the remaining audio using the same end padding as file transcription.

Live decoding is greedy. Hotword beam decoding and diarization remain available
through the file API; the live page labels this distinction. Recognition quality
is that of the existing streaming model.

Only one packet is in flight per browser connection. At most five seconds can
queue in the browser; if inference cannot keep up, recording stops and explicitly
reports that queued audio was not transcribed. The server allows two
simultaneous microphone sessions, each at most 30 minutes, and closes idle sessions
after 60 seconds. Disconnects release the cache. Microphone permission, connection,
and inference errors are shown on the page.

## WebSocket protocol

Connect to `/stream` on the same host. Browser Origin must match Host. WebSocket
reverse proxies must forward the original Host and allow Upgrade. The endpoint
has the same deployment/authentication boundary as the existing ASR API.

1. Server sends `{"type":"ready","sample_rate":16000,"channels":1,"format":"pcm_s16le","max_packet_samples":3200,"decoder":"greedy","backend":"cpu"}`.
2. Client sends binary PCM16, mono, 16 kHz, up to 3,200 samples per message.
3. Server sends `{"type":"ack","audio_ms":200}` after processing each packet.
   An ack includes `text` when a model chunk was decoded. Text is the accumulated
   transcript: replace the displayed text, rather than appending it.
4. Wait for each ack before sending the next packet. To stop, send remaining
   samples, wait for their acks, then send `{"type":"finish"}`.
5. Server sends `{"type":"final","audio_ms":...,"text":"..."}` and closes.
   Failures send `{"type":"error","message":"..."}` before closing where possible.

`GET /mic-config` reports whether the streaming model is available. The existing
multipart `POST /transcribe` contract is unchanged; `mode=streaming` on that route
still consumes a completed WAV and returns one response.

## Validation

```sh
cargo test --locked --no-default-features --features cpu-only,native-streaming
cargo test --locked --features cuda,native-streaming
node --test tests/mic-worklet.test.cjs
# Test a running server with a mono PCM16 / 16 kHz Persian WAV:
python tests/live_smoke.py --url http://localhost:3000 --audio speech.wav
# Optional browser end-to-end test, with Playwright installed:
MIC_TEST_WAV=/absolute/path/speech.wav node tests/mic_browser.cjs
# CHROME_PATH can point to an existing Chrome binary.
```

Verified on 2026-09-25: 12 Rust tests passed in both CPU-only and CUDA feature
builds, including whole-file versus packetized feature parity and CTC boundary
handling. Live final transcripts matched the file route with 200 ms and irregular
19.9375 ms packets. The 8.82-second Persian fixture processed in approximately
0.12 seconds on Stallion CUDA and 1.38 seconds on Stallion CPU, excluding capture
time and the separate reference request. These are warm single-clip smoke timings,
not latency guarantees or recognition-accuracy measurements.

Headless Chrome's simulated microphone, fed the Persian WAV, exercised actual
AudioContext/AudioWorklet capture, WebSocket transport, the CUDA model, partial
text, final flushing, and track cleanup. No physical microphone was accessed.
Windows and macOS launcher changes have not been exercised on their target systems.
