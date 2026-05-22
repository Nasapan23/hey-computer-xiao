from __future__ import annotations

from pathlib import Path
import json
import os
from threading import Lock
from typing import Any


def transcription_sidecar_path(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.stem}.transcript.json")


def load_transcription_sidecar(wav_path: Path) -> dict[str, str] | None:
    sidecar_path = transcription_sidecar_path(wav_path)
    if not sidecar_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return {
        "transcription_status": str(payload.get("transcription_status", "")).strip(),
        "transcript": str(payload.get("transcript", "")).strip(),
        "transcription_language": str(payload.get("transcription_language", "")).strip(),
        "transcription_error": str(payload.get("transcription_error", "")).strip(),
    }


def save_transcription_sidecar(wav_path: Path, payload: dict[str, str]) -> None:
    sidecar_path = transcription_sidecar_path(wav_path)
    sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class WhisperTranscriber:
    def __init__(self) -> None:
        self._model_name = os.getenv("WAKE_WHISPER_MODEL", "tiny.en").strip() or "tiny.en"
        self._device = os.getenv("WAKE_WHISPER_DEVICE", "cpu").strip() or "cpu"
        self._compute_type = os.getenv("WAKE_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
        self._language = os.getenv("WAKE_WHISPER_LANGUAGE", "en").strip()
        self._init_lock = Lock()
        self._transcribe_lock = Lock()
        self._model: Any = None
        self._model_init_error = ""

        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            self._whisper_model_class = None
            self._model_init_error = f"faster-whisper import failed: {exc}"
        else:
            self._whisper_model_class = WhisperModel

    def transcribe_file(self, wav_path: Path) -> dict[str, str]:
        if self._whisper_model_class is None:
            return {
                "transcription_status": "unavailable",
                "transcript": "",
                "transcription_language": "",
                "transcription_error": self._model_init_error or "faster-whisper is not available",
            }

        model = self._ensure_model()
        if model is None:
            return {
                "transcription_status": "unavailable",
                "transcript": "",
                "transcription_language": "",
                "transcription_error": self._model_init_error or "Whisper model could not be initialized",
            }

        transcribe_kwargs: dict[str, Any] = {
            "beam_size": 1,
            "condition_on_previous_text": False,
            "vad_filter": True,
        }
        if self._language:
            transcribe_kwargs["language"] = self._language

        try:
            with self._transcribe_lock:
                segments, info = model.transcribe(str(wav_path), **transcribe_kwargs)
                transcript = " ".join(segment.text.strip() for segment in segments if segment.text and segment.text.strip()).strip()
                if not transcript:
                    fallback_kwargs = dict(transcribe_kwargs)
                    fallback_kwargs["beam_size"] = 3
                    fallback_kwargs["vad_filter"] = False
                    fallback_kwargs.pop("language", None)
                    fallback_kwargs["no_speech_threshold"] = 0.95
                    fallback_kwargs["log_prob_threshold"] = -2.0
                    fallback_segments, fallback_info = model.transcribe(str(wav_path), **fallback_kwargs)
                    fallback_transcript = " ".join(
                        segment.text.strip() for segment in fallback_segments if segment.text and segment.text.strip()
                    ).strip()
                    if fallback_transcript:
                        transcript = fallback_transcript
                        info = fallback_info
        except Exception as exc:
            return {
                "transcription_status": "error",
                "transcript": "",
                "transcription_language": "",
                "transcription_error": str(exc),
            }

        return {
            "transcription_status": "ready" if transcript else "no_speech",
            "transcript": transcript,
            "transcription_language": str(getattr(info, "language", "")),
            "transcription_error": "",
        }

    def _ensure_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._whisper_model_class is None:
            return None

        with self._init_lock:
            if self._model is not None:
                return self._model
            try:
                self._model = self._whisper_model_class(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            except Exception as exc:
                self._model_init_error = str(exc)
                self._model = None

        return self._model


whisper_transcriber = WhisperTranscriber()
