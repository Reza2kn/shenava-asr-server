#!/usr/bin/env bash
# shenava-asr-server — one-command bring-up.
#
#   ./run.sh [server arguments...]
#
# Downloads pinned model assets from Hugging Face, verifies them, builds the
# native Rust feature set, and starts the server. The private Nemotron package
# is enabled automatically when HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is set, or
# when a verified copy is already present in models/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OFFLINE_REVISION="b485a2da4b96087df52319c40a81f95329951a81"
OFFLINE_BASE="https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline/resolve/$OFFLINE_REVISION"
STREAMING_BASE="https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-streaming/resolve/main"
NEMOTRON_REVISION="e3e5de8565a3d5048453f4b905faafc306dc4cd9"
NEMOTRON_BASE="https://huggingface.co/Reza2kn/shenava-nemotron3-rust/resolve/$NEMOTRON_REVISION"

MODEL_DIR="$ROOT/models"
MODEL="$MODEL_DIR/model.onnx"
TOKENS="$MODEL_DIR/tokens.txt"
STREAMING_MODEL="$MODEL_DIR/koochik-streaming.onnx"
NEMOTRON_MODEL="$MODEL_DIR/nemotron3-streaming.onnx"
NEMOTRON_SIDECAR="$MODEL_DIR/nemotron3-streaming.onnx.silence.bin"

OFFLINE_MODEL_SHA="0bfdf9fc3c531f351ad02d7d6b4b309da7ff2f73eb20e2ae167fa12d074c75ac"
TOKENS_SHA="8e192963f6e666dfa5721e5cbd4710bc1ef592460a45f08cefc94b2db16a6954"
STREAMING_MODEL_SHA="c5e7dc34f472e89bd3c48dd6d511482bce455088c5332349ac3afc36f84a3467"
NEMOTRON_MODEL_SHA="9ae7b8ac29138a613e8a7df8c2aca6a29f24efb9f5dbe2b60ba4345593720962"
NEMOTRON_SIDECAR_SHA="cfe7ea16eb73cdf2c67363c5e0d06a201bc10b3bc7d0e1bac1660b8390dbfbb8"

# Do not print this value. It is passed to curl only when downloading the
# private derived Nemotron package.
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
CURL_AUTH=()
if [ -n "$HF_TOKEN" ]; then
  CURL_AUTH=(-H "Authorization: Bearer $HF_TOKEN")
fi

echo "[shenava-asr-server] ensuring model assets..."
mkdir -p "$MODEL_DIR"

verify_sha() {
  local path="$1" expected="$2"
  if command -v sha256sum >/dev/null 2>&1; then
    [ "$(sha256sum "$path" | awk '{print $1}')" = "$expected" ]
  else
    [ "$(shasum -a 256 "$path" | awk '{print $1}')" = "$expected" ]
  fi
}

download_verified() {
  local url="$1" dest="$2" expected="$3"
  if [ -f "$dest" ] && verify_sha "$dest" "$expected"; then
    echo "[shenava-asr-server] verified $dest"
    return
  fi

  mkdir -p "$(dirname "$dest")"
  local partial="$dest.partial"
  echo "[shenava-asr-server] downloading pinned $(basename "$dest")..."
  if ! curl -fL --retry 3 "${CURL_AUTH[@]}" "$url" -o "$partial"; then
    rm -f "$partial"
    echo "[shenava-asr-server] download failed: $url" >&2
    exit 1
  fi
  if ! verify_sha "$partial" "$expected"; then
    rm -f "$partial"
    echo "[shenava-asr-server] SHA-256 verification failed: $dest" >&2
    exit 1
  fi
  mv -f "$partial" "$dest"
}

download_verified "$OFFLINE_BASE/model.onnx" "$MODEL" "$OFFLINE_MODEL_SHA"
download_verified "$OFFLINE_BASE/tokens.txt" "$TOKENS" "$TOKENS_SHA"

ENABLE_STREAMING="${SHENAVA_ENABLE_STREAMING:-1}"
case "$ENABLE_STREAMING" in
  1|true|TRUE|yes|YES|on|ON) ENABLE_STREAMING=1 ;;
  0|false|FALSE|no|NO|off|OFF) ENABLE_STREAMING=0 ;;
  *) echo "[shenava-asr-server] SHENAVA_ENABLE_STREAMING must be true or false" >&2; exit 2 ;;
esac

if [ "$ENABLE_STREAMING" = 1 ]; then
  download_verified "$STREAMING_BASE/model.onnx" "$STREAMING_MODEL" "$STREAMING_MODEL_SHA"
fi

ENABLE_DIARIZATION="${SHENAVA_ENABLE_DIARIZATION:-auto}"
case "$ENABLE_DIARIZATION" in
  1|true|TRUE|yes|YES|on|ON) ENABLE_DIARIZATION=1 ;;
  0|false|FALSE|no|NO|off|OFF) ENABLE_DIARIZATION=0 ;;
  auto)
    if [ -n "$HF_TOKEN" ] || {
      [ -f "$NEMOTRON_MODEL" ] && verify_sha "$NEMOTRON_MODEL" "$NEMOTRON_MODEL_SHA" \
        && [ -f "$NEMOTRON_SIDECAR" ] && verify_sha "$NEMOTRON_SIDECAR" "$NEMOTRON_SIDECAR_SHA";
    }; then
      ENABLE_DIARIZATION=1
    else
      ENABLE_DIARIZATION=0
      echo "[shenava-asr-server] HF_TOKEN not set; Nemotron diarization asset will be skipped."
      echo "[shenava-asr-server] Set HF_TOKEN for diarization, or use SHENAVA_ENABLE_DIARIZATION=0 explicitly."
    fi
    ;;
  *) echo "[shenava-asr-server] SHENAVA_ENABLE_DIARIZATION must be true, false, or auto" >&2; exit 2 ;;
esac

if [ "$ENABLE_DIARIZATION" = 1 ]; then
  if [ -z "$HF_TOKEN" ] && {
    [ ! -f "$NEMOTRON_MODEL" ] || ! verify_sha "$NEMOTRON_MODEL" "$NEMOTRON_MODEL_SHA" \
      || [ ! -f "$NEMOTRON_SIDECAR" ] || ! verify_sha "$NEMOTRON_SIDECAR" "$NEMOTRON_SIDECAR_SHA";
  }; then
    echo "[shenava-asr-server] Nemotron is a private Hugging Face artifact." >&2
    echo "[shenava-asr-server] Export an access token first: HF_TOKEN=hf_... ./run.sh" >&2
    echo "[shenava-asr-server] To run offline ASR only: SHENAVA_ENABLE_DIARIZATION=0 ./run.sh" >&2
    exit 1
  fi
  download_verified "$NEMOTRON_BASE/nemotron3-streaming.onnx" "$NEMOTRON_MODEL" "$NEMOTRON_MODEL_SHA"
  download_verified "$NEMOTRON_BASE/nemotron3-streaming.onnx.silence.bin" "$NEMOTRON_SIDECAR" "$NEMOTRON_SIDECAR_SHA"
fi

echo "[shenava-asr-server] building native Rust release..."
FEATURES="native-diarization,native-streaming"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "[shenava-asr-server] NVIDIA GPU detected — enabling Tract CUDA backend."
  FEATURES="cuda,$FEATURES"
  BACKEND="gpu-or-cpu"
else
  BACKEND="cpu"
fi
cargo build --release --locked --no-default-features --features "$FEATURES"

SERVER_ARGS=(
  --model "$MODEL"
  --tokens "$TOKENS"
  --mel "$ROOT/assets/mel_filters.json"
  --backend "$BACKEND"
)
if [ "$ENABLE_DIARIZATION" = 1 ]; then
  SERVER_ARGS+=(
    --diarizer-nemotron-model "$NEMOTRON_MODEL"
    --diarizer-native-backend "$BACKEND"
  )
fi
if [ "$ENABLE_STREAMING" = 1 ]; then
  SERVER_ARGS+=(
    --streaming-model "$STREAMING_MODEL"
    --streaming-tokens "$TOKENS"
    --streaming-backend "$BACKEND"
  )
fi

echo "[shenava-asr-server] starting on ${ADDR:-0.0.0.0:3000} (backend=$BACKEND, diarization=$ENABLE_DIARIZATION, streaming=$ENABLE_STREAMING)..."
exec ./target/release/shenava-asr-server "${SERVER_ARGS[@]}" "$@"
