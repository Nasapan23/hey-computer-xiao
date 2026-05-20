from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Tuple

import cv2
import numpy as np
import requests


FEATURE_VECTOR_LENGTH = 48 * 48


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_base_url(board_ip_or_url: str) -> str:
    value = board_ip_or_url.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    return f"http://{value}"


def fetch_frame(base_url: str, timeout_s: float = 6.0) -> np.ndarray:
    response = requests.get(f"{base_url}/capture", timeout=timeout_s)
    response.raise_for_status()

    img_buf = np.frombuffer(response.content, dtype=np.uint8)
    frame = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG frame from /capture")
    return frame


def frame_to_fingerprint(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(-1)


def frame_features(frame_bgr: np.ndarray, previous_gray: np.ndarray | None) -> Tuple[Dict[str, float], np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    motion_ratio = 0.0
    if previous_gray is not None:
        diff = cv2.absdiff(gray, previous_gray)
        motion_ratio = float(np.count_nonzero(diff > 18)) / float(diff.size)

    features = {
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "motion_ratio": motion_ratio,
    }
    return features, gray


def post_status(base_url: str, status: str, score: float, timeout_s: float = 3.0) -> None:
    payload = {"status": status, "score": round(float(score), 6), "timestamp": utc_now_iso()}
    response = requests.post(f"{base_url}/status", json=payload, timeout=timeout_s)
    response.raise_for_status()

