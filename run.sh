#!/usr/bin/env bash
# shenava-asr-server — one-command bring-up.
#
#   ./run.sh [--addr 0.0.0.0:3000] [--hotwords hotwords.txt] [extra args...]
#
# Downloads pinned model + token assets from Hugging Face on first run, verifies
# them, builds in release mode, and starts the server. Requires cargo + curl.

set -euo pipefail

REVISION="b485a2da4b96087df52319c40a81f95329951a81"
BASE_URL="https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline/resolve/$REVISION"
MODEL_DIR="models"
MODEL="$MODEL_DIR/model.onnx"
TOKENS="$MODEL_DIR/tokens.txt"

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
  local name="$1"
  local expected="$2"
  local dest="$MODEL_DIR/$name"
  if [ -f "$dest" ] && verify_sha "$dest" "$expected"; then
    echo "[shenava-asr-server] verified $dest"
    return
  fi
  local partial="$dest.partial"
  echo "[shenava-asr-server] downloading pinned $name..."
  curl -fL --retry 3 "$BASE_URL/$name" -o "$partial"
  if ! verify_sha "$partial" "$expected"; then
    rm -f "$partial"
    echo "[shenava-asr-server] SHA-256 verification failed: $name" >&2
    exit 1
  fi
  mv -f "$partial" "$dest"
}

download_verified "model.onnx" "0bfdf9fc3c531f351ad02d7d6b4b309da7ff2f73eb20e2ae167fa12d074c75ac"
download_verified "tokens.txt" "8e192963f6e666dfa5721e5cbd4710bc1ef592460a45f08cefc94b2db16a6954"

echo "[shenava-asr-server] building (release)..."
FEATURES=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "[shenava-asr-server] NVIDIA GPU detected — enabling CUDA backend."
  FEATURES="--features cuda"
fi
cargo build --release $FEATURES

BACKEND="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  BACKEND="gpu-or-cpu"
fi

echo "[shenava-asr-server] starting on ${ADDR:-0.0.0.0:3000} (backend=$BACKEND)..."
exec ./target/release/shenava-asr-server \
  --model "$MODEL" \
  --tokens "$TOKENS" \
  --mel assets/mel_filters.json \
  --backend "$BACKEND" \
  "$@"
