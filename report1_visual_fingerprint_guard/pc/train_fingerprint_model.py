from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA

from common import FEATURE_VECTOR_LENGTH, frame_to_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "normal_frames"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "fingerprint_guard.joblib"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "models" / "training_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCA visual fingerprint anomaly model.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help=f"Directory with normal frames (default: {DEFAULT_FRAMES_DIR})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model output path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary output path (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=25,
        help="PCA components. Lower is faster; higher can be more accurate.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=97.0,
        help="Percentile of training reconstruction errors used as anomaly threshold.",
    )
    parser.add_argument(
        "--threshold-margin",
        type=float,
        default=0.0008,
        help="Small margin added to threshold for stability.",
    )
    return parser.parse_args()


def load_frame_matrix(frames_dir: Path) -> np.ndarray:
    image_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) + list(frames_dir.glob("*.png")))
    if len(image_paths) < 30:
        raise ValueError("Need at least 30 normal frames for a stable baseline.")

    vectors = []
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        vec = frame_to_fingerprint(frame)
        if vec.shape[0] != FEATURE_VECTOR_LENGTH:
            continue
        vectors.append(vec)

    if not vectors:
        raise ValueError("No valid images loaded from frames directory.")

    return np.stack(vectors, axis=0)


def main() -> None:
    args = parse_args()

    matrix = load_frame_matrix(args.frames_dir)
    n_samples = matrix.shape[0]
    n_components = min(args.n_components, n_samples - 1, matrix.shape[1] - 1)
    n_components = max(n_components, 2)

    pca = PCA(n_components=n_components, random_state=42)
    compressed = pca.fit_transform(matrix)
    reconstructed = pca.inverse_transform(compressed)

    errors = np.mean((matrix - reconstructed) ** 2, axis=1)
    threshold = float(np.percentile(errors, args.threshold_percentile) + args.threshold_margin)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pca": pca, "threshold": threshold}, args.model_path)

    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "frames_dir": str(args.frames_dir),
        "samples": int(n_samples),
        "input_vector_length": int(matrix.shape[1]),
        "n_components": int(n_components),
        "threshold_percentile": float(args.threshold_percentile),
        "threshold_margin": float(args.threshold_margin),
        "threshold": float(threshold),
        "train_error_mean": float(np.mean(errors)),
        "train_error_std": float(np.std(errors)),
        "train_error_min": float(np.min(errors)),
        "train_error_max": float(np.max(errors)),
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Samples: {n_samples}")
    print(f"PCA components: {n_components}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Model saved to: {args.model_path}")
    print(f"Summary saved to: {args.summary_path}")


if __name__ == "__main__":
    main()

