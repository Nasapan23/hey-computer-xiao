from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from common import fetch_frame, frame_features, frame_to_fingerprint, normalize_base_url, post_status, utc_now_iso


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "fingerprint_guard.joblib"
DEFAULT_LOG_PATH = PROJECT_ROOT / "report" / "live_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live visual anomaly guard.")
    parser.add_argument("--board-ip", required=True, help="Board IP or URL, e.g. 192.168.1.50")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path for CSV log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument("--interval", type=float, default=0.8, help="Seconds between checks.")
    parser.add_argument("--show", action="store_true", help="Show live OpenCV window.")
    parser.add_argument("--no-post", action="store_true", help="Do not send /status back to board.")
    return parser.parse_args()


def ensure_csv_header(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        return
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_utc",
                "status",
                "score",
                "threshold",
                "brightness_mean",
                "brightness_std",
                "laplacian_var",
                "motion_ratio",
            ]
        )


def main() -> None:
    args = parse_args()
    base_url = normalize_base_url(args.board_ip)

    bundle = joblib.load(args.model_path)
    pca = bundle["pca"]
    threshold = float(bundle["threshold"])

    ensure_csv_header(args.log_path)
    prev_gray = None
    last_status = None

    print(f"Live guard started. Base URL: {base_url}")
    print(f"Using threshold: {threshold:.8f}")
    print("Press q in window to exit (if --show is used).")

    while True:
        timestamp = utc_now_iso()
        frame = fetch_frame(base_url)
        vec = frame_to_fingerprint(frame).reshape(1, -1)

        reconstruction = pca.inverse_transform(pca.transform(vec))
        error = float(np.mean((vec - reconstruction) ** 2))
        status = "ABNORMAL" if error > threshold else "NORMAL"

        features, prev_gray = frame_features(frame, prev_gray)

        if (not args.no_post) and (status != last_status):
            try:
                post_status(base_url, status, error)
            except Exception as exc:
                print(f"Status POST failed: {exc}")

        with args.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    status,
                    f"{error:.8f}",
                    f"{threshold:.8f}",
                    f"{features['brightness_mean']:.3f}",
                    f"{features['brightness_std']:.3f}",
                    f"{features['laplacian_var']:.3f}",
                    f"{features['motion_ratio']:.6f}",
                ]
            )

        print(
            f"{timestamp} | {status:<8} | score={error:.8f} "
            f"| mean={features['brightness_mean']:.1f} motion={features['motion_ratio']:.4f}"
        )

        if args.show:
            overlay = frame.copy()
            color = (0, 200, 0) if status == "NORMAL" else (0, 0, 255)
            cv2.putText(
                overlay,
                f"{status} score={error:.6f} thr={threshold:.6f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("XIAO Visual Fingerprint Guard", overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        last_status = status
        time.sleep(args.interval)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

