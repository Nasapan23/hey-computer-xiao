from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import (
    MAX_GAIN,
    MIN_GAIN,
    REMOVE_DC_OFFSET,
    SAMPLE_RATE,
    STRIDE_SAMPLES,
    TARGET_RMS,
    WINDOW_SAMPLES,
)


def load_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    original_rate, audio = wavfile.read(path)

    original_dtype = audio.dtype
    is_integer_pcm = np.issubdtype(original_dtype, np.integer)
    max_value = float(np.iinfo(original_dtype).max) if is_integer_pcm else 1.0

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if is_integer_pcm:
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        audio = signal.resample_poly(
            audio,
            sample_rate // divisor,
            original_rate // divisor,
        ).astype(np.float32)

    return np.clip(audio, -1.0, 1.0)


def slice_windows(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < WINDOW_SAMPLES:
        padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded[None, :], np.array([0], dtype=np.int64)

    windows = []
    starts = []
    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        windows.append(audio[start : start + WINDOW_SAMPLES])
        starts.append(start)

    if not windows:
        windows.append(audio[:WINDOW_SAMPLES])
        starts.append(0)

    return np.stack(windows).astype(np.float32), np.array(starts, dtype=np.int64)


def choose_windows(
    windows: np.ndarray,
    starts: np.ndarray,
    source_label: str,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows) <= max_windows:
        return windows, starts

    if source_label == "background_noise":
        indices = np.linspace(0, len(windows) - 1, max_windows).round().astype(int)
        return windows[indices], starts[indices]

    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    indices = np.argsort(rms)[-max_windows:]
    indices.sort()
    return windows[indices], starts[indices]


def preprocess_windows(windows: np.ndarray, apply_rms_normalization: bool) -> np.ndarray:
    out = windows.copy()
    for i in range(len(out)):
        window = out[i]
        if REMOVE_DC_OFFSET:
            window = window - np.mean(window, dtype=np.float32)

        if apply_rms_normalization:
            rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float32)))
            if rms > 1e-8:
                gain = np.clip(TARGET_RMS / rms, MIN_GAIN, MAX_GAIN)
                window = window * gain

        out[i] = np.clip(window, -1.0, 1.0)
    return out


def write_preview_windows(
    processed_dir: Path,
    wav_path: Path,
    label: str,
    windows: np.ndarray,
    starts: np.ndarray,
) -> None:
    output_dir = processed_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in wav_path.stem)
    for index, (window, start) in enumerate(zip(windows, starts), start=1):
        start_ms = int(round(start * 1000 / SAMPLE_RATE))
        output_path = output_dir / f"{safe_stem}_window_{index:02d}_{start_ms}ms.wav"
        int16_audio = np.clip(window * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(output_path, SAMPLE_RATE, int16_audio)

