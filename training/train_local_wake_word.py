from __future__ import annotations

"""CLI entry point for local TinyML wake-word training.

The implementation lives in training/wake_word/ so the project is easier to
present and each file has one clear responsibility.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local TinyML wake-word model from WAV files.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("training/output"))
    parser.add_argument("--header-output", type=Path, default=Path("esp32_tinyml_wake_word") / "model_data.h")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows-per-file", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--normalize-rms",
        action="store_true",
        help="Apply per-window RMS normalization (usually off for tiny XIAO-only datasets).",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from wake_word.pipeline import train_and_export
    except ImportError:
        from training.wake_word.pipeline import train_and_export

    train_and_export(parse_args())


if __name__ == "__main__":
    main()
