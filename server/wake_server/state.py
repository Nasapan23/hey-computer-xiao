from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

from .config import DETECTION_COUNTER_LABELS


class ServerState:
    def __init__(self) -> None:
        self._counter_lock = Lock()
        self._collect_counter_lock = Lock()
        self._activity_lock = Lock()
        self._authorized_wake_count = 0
        self._detection_counters = {label: 0 for label in DETECTION_COUNTER_LABELS}
        self._collect_counter = 0
        self._signal_events: deque[dict[str, Any]] = deque(maxlen=500)
        self._command_uploads: deque[dict[str, Any]] = deque(maxlen=500)

    def next_collect_number(self) -> int:
        with self._collect_counter_lock:
            self._collect_counter += 1
            return self._collect_counter

    def record_detection(self, event: str, authorized: bool, score: str, timestamp: str) -> tuple[int, int]:
        with self._counter_lock:
            if event in self._detection_counters:
                self._detection_counters[event] += 1
            if authorized or event == "authorized_user_wake":
                self._authorized_wake_count += 1
            authorized_wake_count = self._authorized_wake_count
            event_count = self._detection_counters.get(event, 0)

        with self._activity_lock:
            self._signal_events.appendleft(
                {
                    "timestamp": timestamp,
                    "event": event,
                    "authorized": bool(authorized),
                    "score": score,
                    "authorized_wake_count": authorized_wake_count,
                    "event_count": event_count,
                }
            )

        return authorized_wake_count, event_count

    def record_command_upload(
        self,
        timestamp: str,
        trigger_label: str,
        trigger_score: str,
        relative_wav_path: str,
        sample_rate: int,
        byte_count: int,
    ) -> None:
        with self._activity_lock:
            self._command_uploads.appendleft(
                {
                    "timestamp": timestamp,
                    "trigger_label": trigger_label,
                    "trigger_score": trigger_score,
                    "path": relative_wav_path,
                    "sample_rate": sample_rate,
                    "bytes": byte_count,
                    "transcription_status": "pending",
                    "transcript": "",
                    "transcription_language": "",
                    "transcription_error": "",
                }
            )

    def set_command_transcription(
        self,
        relative_wav_path: str,
        status: str,
        transcript: str,
        language: str = "",
        error: str = "",
    ) -> None:
        with self._activity_lock:
            for entry in self._command_uploads:
                if entry.get("path") != relative_wav_path:
                    continue
                entry["transcription_status"] = status
                entry["transcript"] = transcript
                entry["transcription_language"] = language
                entry["transcription_error"] = error
                break

    def read_recent_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._activity_lock:
            return list(self._signal_events)[: max(1, limit)]

    def read_recent_command_uploads(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._activity_lock:
            return list(self._command_uploads)[: max(1, limit)]

    def read_counters(self) -> dict[str, int | dict[str, int]]:
        with self._counter_lock:
            return {
                "authorized_wake_count": self._authorized_wake_count,
                "events": dict(self._detection_counters),
            }


state = ServerState()
