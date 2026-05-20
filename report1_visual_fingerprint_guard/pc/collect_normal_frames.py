from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from common import fetch_frame, frame_features, normalize_base_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "normal_frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NORMAL scene frames from XIAO camera.")
    parser.add_argument("--board-ip", required=True, help="Board IP or URL, e.g. 192.168.1.50")
    parser.add_argument("--samples", type=int, default=150, help="Number of frames to collect.")
    parser.add_argument("--interval", type=float, default=0.7, help="Seconds between samples.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help=f"Where to save frames (default: {DEFAULT_FRAMES_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = normalize_base_url(args.board_ip)

    args.frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.samples} normal frames from {base_url}")
    print(f"Saving frames to: {args.frames_dir}")

    prev_gray = None
    for idx in range(args.samples):
        frame = fetch_frame(base_url)
        features, prev_gray = frame_features(frame, prev_gray)

        out_name = f"normal_{idx:04d}.jpg"
        out_path = args.frames_dir / out_name
        cv2.imwrite(str(out_path), frame)

        print(
            f"[{idx + 1:03d}/{args.samples}] saved {out_name} | "
            f"mean={features['brightness_mean']:.1f} std={features['brightness_std']:.1f} "
            f"motion={features['motion_ratio']:.4f}"
        )

        if idx + 1 < args.samples:
            time.sleep(args.interval)

    print("Collection complete.")


if __name__ == "__main__":
    main()

