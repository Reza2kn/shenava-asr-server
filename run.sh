#!/usr/bin/env bash
# shenava-asr-server — one-command bring-up.
#
#   ./run.sh [server arguments...]
#
# Downloads pinned model assets from Hugging Face, verifies them, builds the
# native Rust feature set, and starts the server. The public Nemotron package
# is downloaded automatically; HF_TOKEN is accepted but not required.

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

# Do not print this value. It is passed to curl only for Hugging Face
# downloads. The public packages do not require it, but it also works for
# installations using a private mirror.
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
  auto) ENABLE_DIARIZATION=1 ;;
  *) echo "[shenava-asr-server] SHENAVA_ENABLE_DIARIZATION must be true, false, or auto" >&2; exit 2 ;;
esac

if [ "$ENABLE_DIARIZATION" = 1 ]; then
  download_verified "$NEMOTRON_BASE/nemotron3-streaming.onnx" "$NEMOTRON_MODEL" "$NEMOTRON_MODEL_SHA"
  download_verified "$NEMOTRON_BASE/nemotron3-streaming.onnx.silence.bin" "$NEMOTRON_SIDECAR" "$NEMOTRON_SIDECAR_SHA"
fi

echo "[shenava-asr-server] building native Rust release..."
FEATURES="native-diarization,native-streaming"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "[shenava-asr-server] NVIDIA GPU detected — requiring the Tract CUDA backend."
  FEATURES="cuda,$FEATURES"
  BACKEND="cuda"

  # NVIDIA's pip wheels intentionally install cuDNN outside the system loader
  # path. Locate only the cuDNN directory: adding an entire Python environment
  # can accidentally mix its cuBLAS/cuBLASLt with /usr/local/cuda.
  CUDNN_VISIBLE=0
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libcudnn\.so'; then
    CUDNN_VISIBLE=1
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import ctypes; ctypes.CDLL("libcudnn.so.9")' >/dev/null 2>&1; then
    CUDNN_VISIBLE=1
  fi
  if [ "$CUDNN_VISIBLE" = 0 ]; then
    CUDNN_LIB_DIR=""
    if command -v python3 >/dev/null 2>&1; then
      CUDNN_LIB_DIR="$(python3 - <<'PY'
import glob
import os
import site
import sysconfig

candidates = []
venv = os.environ.get("VIRTUAL_ENV")
if venv:
    candidates.extend(glob.glob(os.path.join(venv, "lib/python*/site-packages/nvidia/cudnn/lib")))
for root in site.getsitepackages() + [site.getusersitepackages(), sysconfig.get_path("purelib")]:
    if root:
        candidates.append(os.path.join(root, "nvidia/cudnn/lib"))
candidates.extend(glob.glob(os.path.expanduser("~/.local/lib/python*/site-packages/nvidia/cudnn/lib")))
candidates.extend(glob.glob("/usr/local/lib/python*/dist-packages/nvidia/cudnn/lib"))
candidates.extend(glob.glob("/usr/local/lib/python*/site-packages/nvidia/cudnn/lib"))
candidates.extend(glob.glob("/opt/*/lib/python*/site-packages/nvidia/cudnn/lib"))
for path in candidates:
    if glob.glob(os.path.join(path, "libcudnn.so*")):
        print(path)
        break
PY
)"
    fi
    if [ -n "$CUDNN_LIB_DIR" ]; then
      export LD_LIBRARY_PATH="$CUDNN_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      echo "[shenava-asr-server] found cuDNN in $CUDNN_LIB_DIR"
    else
      echo "[shenava-asr-server] cuDNN is not visible to the dynamic loader." >&2
      echo "[shenava-asr-server] Install NVIDIA cuDNN or add its lib directory to LD_LIBRARY_PATH." >&2
      echo "[shenava-asr-server] For pip: python3 -m pip install nvidia-cudnn-cu13" >&2
      exit 1
    fi
  fi
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
