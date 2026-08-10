#!/usr/bin/env bash
# shenava-asr-server — one-command bring-up.
#
#   ./run.sh [--addr 0.0.0.0:3000] [--hotwords hotwords.txt] [extra args...]
#
# Downloads the model + tokens from Hugging Face on first run, builds in release
# mode, and starts the server. Requires only Rust (cargo) and network access.

set -euo pipefail

REPO="Reza2kn/Shenava-Koochik-v1.0-tract-offline"
MODEL_DIR="models"
MODEL="$MODEL_DIR/model.onnx"
TOKENS="$MODEL_DIR/tokens.txt"

echo "[shenava-asr-server] ensuring model assets..."
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL" ] || [ ! -s "$MODEL" ]; then
  echo "[shenava-asr-server] downloading $REPO/model.onnx (~418 MB, first run only)..."
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" model.onnx --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
  else
    python3 - "$REPO" "$MODEL_DIR" <<'PY'
import sys, os
repo, d = sys.argv[1], sys.argv[2]
try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("huggingface_hub not installed; run: pip install -U huggingface_hub")
p = snapshot_download(repo_id=repo, local_dir=d, allow_patterns=["model.onnx", "tokens.txt"])
print("downloaded to", p)
PY
  fi
  echo "[shenava-asr-server] model downloaded."
else
  echo "[shenava-asr-server] model present."
fi

if [ ! -f "$TOKENS" ] || [ ! -s "$TOKENS" ]; then
  echo "[shenava-asr-server] downloading tokens.txt..."
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" tokens.txt --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
  else
    python3 - "$REPO" "$MODEL_DIR" <<'PY'
import sys
repo, d = sys.argv[1], sys.argv[2]
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=repo, filename="tokens.txt", local_dir=d)
print("tokens.txt downloaded")
PY
  fi
fi

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
