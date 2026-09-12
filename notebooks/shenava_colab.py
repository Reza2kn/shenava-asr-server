"""Small, Colab-safe helpers for the three Shenava v1 sherpa-onnx exports."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download


MODELS = {
    "koochik": {
        "title": "Shenava Koochik v1.0 (114M)",
        "repo": "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx",
    },
    "rizeh": {
        "title": "Shenava Rizeh v1.0 (32M)",
        "repo": "Reza2kn/Shenava-Rizeh-v1.0-sherpa-onnx",
    },
    "rizeh-pizeh": {
        "title": "Shenava Rizeh-Pizeh v1.0 (6.9M)",
        "repo": "Reza2kn/Shenava-Rizeh-Pizeh-v1.0-sherpa-onnx",
    },
}

HOTWORDS_URL = (
    "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx/"
    "resolve/main/hotwords_fa.txt"
)
HOTWORDS_SHA256 = "7a8606b654bb23aa24db9774d1ca4feab06757677621adb6853f1ba5ed4affd9"
MEL_URL = (
    "https://raw.githubusercontent.com/Reza2kn/shenava-asr-server/main/"
    "assets/mel_filters.json"
)

_recognizer = None
_recognizer_key = None


def _asset_dir() -> Path:
    return Path(os.environ.get("SHENAVA_CACHE_DIR", "/content/shenava-assets"))


def _download_url(url: str, destination: Path) -> Path:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    return destination


def model_dir(key: str) -> Path:
    if key not in MODELS:
        raise KeyError(f"Unknown model {key!r}; choose from {list(MODELS)}")
    cfg = MODELS[key]
    return Path(
        snapshot_download(
            cfg["repo"],
            allow_patterns=["model.onnx", "tokens.txt", "persian_itn.py"],
        )
    )


def load_recognizer(key: str, num_threads: int = 1):
    """Keep only one native recognizer alive, avoiding Colab memory spikes."""
    global _recognizer, _recognizer_key
    if _recognizer is not None and _recognizer_key == key:
        return _recognizer

    _recognizer = None
    _recognizer_key = None
    gc.collect()

    import sherpa_onnx

    folder = model_dir(key)
    _recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(folder / "model.onnx"),
        tokens=str(folder / "tokens.txt"),
        num_threads=max(1, int(num_threads)),
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )
    _recognizer_key = key
    return _recognizer


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all() or audio.size == 0:
        raise ValueError("Audio is empty or contains invalid samples")
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def transcribe_greedy(key: str, audio_path: str | Path, num_threads: int = 1) -> str:
    audio, sample_rate = read_audio(audio_path)
    recognizer = load_recognizer(key, num_threads=num_threads)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def _fbank(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """NeMo-compatible, unnormalized 80-bin log-mel features used by Shenava."""
    if sample_rate != 16000:
        output_count = int(np.ceil(len(audio) * 16000 / sample_rate))
        old_x = np.arange(len(audio), dtype=np.float64)
        new_x = np.arange(output_count, dtype=np.float64) * sample_rate / 16000
        audio = np.interp(new_x, old_x, audio).astype(np.float32)

    pre = np.empty_like(audio)
    pre[0] = audio[0]
    pre[1:] = audio[1:] - np.float32(0.97) * audio[:-1]
    padded = np.pad(pre, (256, 256), mode="reflect")
    frame_count = 1 + len(audio) // 160
    starts = np.arange(frame_count)[:, None] * 160
    frames = padded[np.minimum(starts + np.arange(512)[None, :], len(padded) - 1)]

    window = np.zeros(512, dtype=np.float32)
    window[56:456] = np.hanning(400).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window[None, :], n=512, axis=1)) ** 2

    mel_path = _download_url(MEL_URL, _asset_dir() / "mel_filters.json")
    mel = np.asarray(json.loads(mel_path.read_text()), dtype=np.float32)
    return np.log(power.astype(np.float32) @ mel.T + np.float32(2**-24)).T.astype(np.float32)


def _labels(tokens_path: Path) -> list[str]:
    rows = []
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        token, idx = line.rsplit(maxsplit=1)
        rows.append((int(idx), token))
    rows.sort()
    labels = []
    for idx, token in rows:
        if token == "<blk>":
            labels.append("")
        elif token.startswith("<") and token.endswith(">"):
            labels.append(chr(0xE000 + idx))
        else:
            labels.append(token)
    return labels


def _hotwords() -> list[str]:
    path = _download_url(HOTWORDS_URL, _asset_dir() / "hotwords_fa.txt")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != HOTWORDS_SHA256:
        raise RuntimeError(f"Hotword checksum mismatch: {digest}")
    words = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    words = [x for x in words if x and not x.startswith("#")]
    if len(words) != 3669:
        raise RuntimeError(f"Expected 3,669 hotwords, found {len(words):,}")
    return words


def _logaddexp(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + float(np.log1p(np.exp(lo - hi)))


def _ctc_hotbeam(
    log_probs: np.ndarray,
    labels: list[str],
    hotword_lines: list[str],
    beam_width: int,
    hotword_weight: float,
) -> str:
    """Python port of shenava-ctc-beam's no-LM prefix beam."""
    blank_id = labels.index("")
    words = [word for line in hotword_lines for word in line.split() if word]
    word_set = set(words)
    shortest_prefix: dict[str, int] = {}
    for word in words:
        length = len(word)
        for i in range(1, length + 1):
            prefix = word[:i]
            shortest_prefix[prefix] = min(shortest_prefix.get(prefix, length), length)

    def partial(word_part: str) -> float:
        denominator = shortest_prefix.get(word_part)
        return 0.0 if not denominator else hotword_weight * len(word_part) / denominator

    beams = {("", "", -1): (0, 0.0)}
    for frame in np.asarray(log_probs, dtype=np.float32):
        candidates = np.flatnonzero(frame >= -5.0).tolist()
        argmax = int(frame.argmax())
        if argmax not in candidates:
            candidates.append(argmax)
        merged: dict[tuple[str, str, int], tuple[int, float]] = {}
        for idx in candidates:
            token = labels[idx]
            for (text, part, last_idx), (count, score) in beams.items():
                if idx == blank_id or idx == last_idx:
                    key, new_count = (text, part, idx), count
                elif token.startswith("▁"):
                    if part:
                        text = f"{text} {part}".strip()
                        count += int(part in word_set)
                    key = (text, token.strip("▁"), idx)
                    new_count = count
                else:
                    key, new_count = (text, part + token, idx), count
                new_score = score + float(frame[idx])
                if key in merged:
                    old_count, old_score = merged[key]
                    merged[key] = (old_count, _logaddexp(old_score, new_score))
                else:
                    merged[key] = (new_count, new_score)
        scored = []
        for key, (count, score) in merged.items():
            rank = score + hotword_weight * count + partial(key[1])
            scored.append((rank, key, count, score))
        best = max(item[0] for item in scored)
        scored = [item for item in scored if item[0] >= best - 10.0]
        scored.sort(key=lambda item: item[0], reverse=True)
        beams = {key: (count, score) for _, key, count, score in scored[:beam_width]}

    winner, winner_score = "", -np.inf
    for (text, part, _), (count, score) in beams.items():
        if part:
            text = f"{text} {part}".strip()
            count += int(part in word_set)
        final_score = score + hotword_weight * count
        if final_score > winner_score:
            winner, winner_score = text, final_score
    return "".join(c for c in winner if not "\ue000" <= c <= "\ue0ff").strip()


def transcribe_hotbeam(
    key: str,
    audio_path: str | Path,
    beam_width: int = 80,
    hotword_weight: float = 2.5,
) -> str:
    """Run Shenava's 3,669-word CTC hotword beam over the model logits."""
    import onnxruntime as ort
    folder = model_dir(key)
    audio, sample_rate = read_audio(audio_path)
    features = _fbank(audio, sample_rate)[None, :, :]
    length = np.asarray([features.shape[2]], dtype=np.int64)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(folder / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    logits, output_length = session.run(
        None, {"audio_signal": features, "length": length}
    )
    return _ctc_hotbeam(
        logits[0, : int(output_length[0])],
        _labels(folder / "tokens.txt"),
        _hotwords(),
        int(beam_width),
        float(hotword_weight),
    )


def upload_and_transcribe(key: str, include_hotbeam: bool = False) -> None:
    """Colab-facing one-click upload and transcription flow."""
    from google.colab import files

    print(f"Loading {MODELS[key]['title']} …")
    load_recognizer(key)
    print("Choose a WAV, FLAC, MP3, M4A, or OGG recording:")
    uploaded = files.upload()
    if not uploaded:
        print("No file selected.")
        return
    for filename, payload in uploaded.items():
        path = Path("/content") / Path(filename).name
        path.write_bytes(payload)
        print(f"\n{path.name}")
        print("Greedy:", transcribe_greedy(key, path))
        if include_hotbeam:
            print("Hotbeam (3,669 words):", transcribe_hotbeam(key, path))
