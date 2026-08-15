#!/usr/bin/env bash
# Native CoreML bring-up for macOS 13+ (Apple Silicon recommended).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COREML_REV="14cefcd170e8307e5d65785e172e17453b25873f"
TRACT_REV="b485a2da4b96087df52319c40a81f95329951a81"
COREML_BASE="https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16/resolve/$COREML_REV"
TOKEN_URL="https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline/resolve/$TRACT_REV/tokens.txt"
PACKAGE="$ROOT/models/coreml/shenava-koochik-v1.0_ctc_fixed2005_len_fp16.mlpackage"

download_verified() {
  local url="$1" dest="$2" expected="$3"
  if [ -f "$dest" ] && [ "$(shasum -a 256 "$dest" | awk '{print $1}')" = "$expected" ]; then
    echo "[shenava] verified $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  local partial="$dest.partial"
  curl -fL --retry 3 "$url" -o "$partial"
  if [ "$(shasum -a 256 "$partial" | awk '{print $1}')" != "$expected" ]; then
    rm -f "$partial"
    echo "[shenava] SHA-256 verification failed for $dest" >&2
    exit 1
  fi
  mv -f "$partial" "$dest"
}

download_verified "$COREML_BASE/shenava-koochik-v1.0_ctc_fixed2005_len_fp16.mlpackage/Manifest.json" \
  "$PACKAGE/Manifest.json" "f0c7cf8c88d842c6631534bce7629e307acb7039379b499c7bbf2b1010acfbb8"
download_verified "$COREML_BASE/shenava-koochik-v1.0_ctc_fixed2005_len_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel" \
  "$PACKAGE/Data/com.apple.CoreML/model.mlmodel" "e04b16f56e0648073b7eb22f9050ad838bc2553795f397782967dd04e37e36ab"
download_verified "$COREML_BASE/shenava-koochik-v1.0_ctc_fixed2005_len_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin" \
  "$PACKAGE/Data/com.apple.CoreML/weights/weight.bin" "664ff3f2fa0b37229a99a240f9d5ba375df84a8d33ec9a67a93d1c0279a4e8d4"
download_verified "$TOKEN_URL" "$ROOT/models/tokens.txt" \
  "8e192963f6e666dfa5721e5cbd4710bc1ef592460a45f08cefc94b2db16a6954"

cd "$ROOT"
cargo build --release --locked --no-default-features --features coreml
exec ./target/release/shenava-asr-server \
  --model "$PACKAGE" \
  --tokens "$ROOT/models/tokens.txt" \
  --mel "$ROOT/assets/mel_filters.json" \
  --backend coreml \
  "$@"
