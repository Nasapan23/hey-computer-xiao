from __future__ import annotations

from pathlib import Path

RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "recordings"
DATASET_RAW_DIR = Path(__file__).resolve().parents[2] / "dataset" / "raw"

LABEL_TO_DATASET_DIR = {
    "authorized_user_wake": "authorized_user_wake",
    "wake": "authorized_user_wake",
    "unknown_user_wake": "unknown_user_wake",
    "not_wake": "unknown_speech",
}

DETECTION_COUNTER_LABELS = ("authorized_user_wake", "unknown_user_wake")

