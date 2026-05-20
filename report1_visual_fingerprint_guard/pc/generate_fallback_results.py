from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

from common import frame_features, frame_to_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMAL_DIR = PROJECT_ROOT / "data" / "normal_frames_fallback"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "fingerprint_guard_fallback.joblib"
SUMMARY_PATH = PROJECT_ROOT / "data" / "models" / "training_summary_fallback.json"
LIVE_RESULTS_PATH = PROJECT_ROOT / "report" / "live_results_fallback.csv"
METRICS_PATH = PROJECT_ROOT / "report" / "fallback_metrics.json"


def make_base_scene(t: int, rng: np.random.Generator, h: int = 240, w: int = 320) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    grad = 95 + 28 * x + 18 * y
    vignette = 1.0 - 0.25 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    img = grad * vignette
    img += 3.0 * np.sin(2 * np.pi * (x * 2.2 + t * 0.01))
    img += 2.0 * np.cos(2 * np.pi * (y * 1.8 + t * 0.008))
    img += rng.normal(0, 1.5, (h, w))
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Static scene objects (desk-like baseline)
    cv2.rectangle(bgr, (30, 150), (130, 220), (110, 110, 110), -1)
    cv2.rectangle(bgr, (180, 140), (300, 210), (140, 140, 140), -1)
    cv2.circle(bgr, (260, 70), 28, (125, 125, 125), -1)
    return bgr


def generate_training_frames(rng: np.random.Generator) -> None:
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    for p in NORMAL_DIR.glob("*.jpg"):
        p.unlink()

    for i in range(180):
        frame = make_base_scene(i, rng)
        alpha = 1.0 + rng.normal(0, 0.01)
        beta = rng.normal(0, 2.0)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        cv2.imwrite(str(NORMAL_DIR / f"normal_{i:04d}.jpg"), frame)


def train_model() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pc" / "train_fingerprint_model.py"),
        "--frames-dir",
        str(NORMAL_DIR),
        "--model-path",
        str(MODEL_PATH),
        "--summary-path",
        str(SUMMARY_PATH),
        "--n-components",
        "25",
        "--threshold-percentile",
        "97",
        "--threshold-margin",
        "0.0008",
    ]
    subprocess.run(cmd, check=True)


def evaluate(rng: np.random.Generator) -> None:
    bundle = joblib.load(MODEL_PATH)
    pca = bundle["pca"]
    threshold = float(bundle["threshold"])

    rows = []
    prev_gray = None
    for i in range(30):
        frame = make_base_scene(300 + i, rng)
        true_label = "NORMAL"

        if i in {5, 6, 7, 12, 18, 19, 24, 25, 28}:
            true_label = "ABNORMAL"
            kind = i % 3
            if kind == 0:
                cv2.rectangle(frame, (120, 40), (220, 120), (255, 255, 255), -1)
            elif kind == 1:
                cv2.rectangle(frame, (0, 0), (320, 35), (30, 30, 30), -1)
            else:
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=45)
        else:
            frame = cv2.convertScaleAbs(frame, alpha=1.0 + rng.normal(0, 0.008), beta=rng.normal(0, 1.5))

        vec = frame_to_fingerprint(frame).reshape(1, -1)
        recon = pca.inverse_transform(pca.transform(vec))
        score = float(np.mean((vec - recon) ** 2))
        pred = "ABNORMAL" if score > threshold else "NORMAL"
        features, prev_gray = frame_features(frame, prev_gray)

        rows.append(
            {
                "index": i + 1,
                "true_label": true_label,
                "predicted_label": pred,
                "score": score,
                "threshold": threshold,
                "brightness_mean": features["brightness_mean"],
                "brightness_std": features["brightness_std"],
                "laplacian_var": features["laplacian_var"],
                "motion_ratio": features["motion_ratio"],
            }
        )

    LIVE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "true_label",
                "predicted_label",
                "score",
                "threshold",
                "brightness_mean",
                "brightness_std",
                "laplacian_var",
                "motion_ratio",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["index"],
                    r["true_label"],
                    r["predicted_label"],
                    f"{r['score']:.8f}",
                    f"{r['threshold']:.8f}",
                    f"{r['brightness_mean']:.3f}",
                    f"{r['brightness_std']:.3f}",
                    f"{r['laplacian_var']:.3f}",
                    f"{r['motion_ratio']:.6f}",
                ]
            )

    tp = sum(1 for r in rows if r["true_label"] == "ABNORMAL" and r["predicted_label"] == "ABNORMAL")
    tn = sum(1 for r in rows if r["true_label"] == "NORMAL" and r["predicted_label"] == "NORMAL")
    fp = sum(1 for r in rows if r["true_label"] == "NORMAL" and r["predicted_label"] == "ABNORMAL")
    fn = sum(1 for r in rows if r["true_label"] == "ABNORMAL" and r["predicted_label"] == "NORMAL")

    metrics = {
        "samples": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(rows),
        "threshold": threshold,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    rng = np.random.default_rng(42)
    generate_training_frames(rng)
    train_model()
    evaluate(rng)
    print("Fallback report data generated.")


if __name__ == "__main__":
    main()

