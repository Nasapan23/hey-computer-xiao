from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import choose_windows, load_wav, preprocess_windows, slice_windows, write_preview_windows
from .config import LABEL_SOURCE_DIRS, LABELS, LEGACY_IGNORED_SOURCE_DIRS, SAMPLE_RATE


def make_dataset(
    dataset_dir: Path,
    max_windows_per_file: int,
    apply_rms_normalization: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Build window-level dataset and manifest from source WAV recordings."""
    for legacy_dir in LEGACY_IGNORED_SOURCE_DIRS:
        legacy_path = dataset_dir / "raw" / legacy_dir
        if not legacy_path.exists():
            continue
        legacy_wavs = list(legacy_path.rglob("*.wav"))
        if legacy_wavs:
            print(
                f"Warning: ignoring {len(legacy_wavs)} legacy WAV files in "
                f"{legacy_path.as_posix()} (not used in current 3-class training)."
            )

    examples = []
    targets = []
    sources = []
    manifest = []

    for label_index, label in enumerate(LABELS):
        wav_paths: list[tuple[str, Path]] = []

        for source_label in LABEL_SOURCE_DIRS[label]:
            source_dir = dataset_dir / "raw" / source_label
            source_paths = sorted(
                path
                for path in source_dir.rglob("*.wav")
                if "camera" not in [part.lower() for part in path.parts]
            )
            wav_paths.extend((source_label, path) for path in source_paths)

        if not wav_paths:
            source_dirs = ", ".join(LABEL_SOURCE_DIRS[label])
            raise RuntimeError(f"No WAV files found for {label}. Checked: {source_dirs}")

        for source_label, wav_path in wav_paths:
            audio = load_wav(wav_path, SAMPLE_RATE)
            windows, starts = slice_windows(audio)
            selected, selected_starts = choose_windows(windows, starts, source_label, max_windows_per_file)
            selected = preprocess_windows(selected, apply_rms_normalization)
            write_preview_windows(dataset_dir / "processed", wav_path, label, selected, selected_starts)

            examples.append(selected)
            targets.extend([label_index] * len(selected))
            sources.extend([str(wav_path)] * len(selected))
            manifest.append(
                {
                    "file": str(wav_path),
                    "label": label,
                    "source_label": source_label,
                    "duration_seconds": round(len(audio) / SAMPLE_RATE, 3),
                    "windows_used": int(len(selected)),
                    "selected_start_seconds": [
                        round(float(start) / SAMPLE_RATE, 3) for start in selected_starts
                    ],
                }
            )

    x = np.concatenate(examples, axis=0)
    y = np.array(targets, dtype=np.int64)
    source_ids = np.array(sources)
    return x, y, source_ids, manifest


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    source_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by source file so windows from one file never leak across splits."""
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    for label_index in range(len(LABELS)):
        label_indices = np.where(y == label_index)[0]
        label_sources = np.unique(source_ids[label_indices])
        rng.shuffle(label_sources)

        if len(label_sources) < 2:
            test_sources = set()
        else:
            test_file_count = max(1, int(round(len(label_sources) * 0.2)))
            test_sources = set(label_sources[:test_file_count])

        for index in label_indices:
            if source_ids[index] in test_sources:
                test_mask[index] = True
            else:
                train_mask[index] = True

    return x[train_mask], y[train_mask], x[test_mask], y[test_mask]
