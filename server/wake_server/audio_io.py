from __future__ import annotations

from pathlib import Path
import wave

from fastapi import HTTPException


def validate_pcm_headers(sample_rate: int, bits_per_sample: int, channels: int) -> None:
    if bits_per_sample != 16:
        raise HTTPException(status_code=400, detail="Only 16-bit PCM is supported")
    if channels != 1:
        raise HTTPException(status_code=400, detail="Only mono PCM is supported")
    if sample_rate <= 0:
        raise HTTPException(status_code=400, detail="Invalid sample rate")


def trim_pcm_to_int16(pcm: bytes) -> bytes:
    if len(pcm) % 2 != 0:
        return pcm[:-1]
    return pcm


def write_wav(path: Path, pcm: bytes, sample_rate: int, bits_per_sample: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def safe_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.strip().lower())
    return cleaned or "unknown"

