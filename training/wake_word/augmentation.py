from __future__ import annotations

import numpy as np

from .config import LABELS, SAMPLE_RATE


def augment_sample(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = sample.copy()

    augmented *= rng.uniform(0.8, 1.2)

    max_shift = int(0.1 * SAMPLE_RATE)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    augmented = np.roll(augmented, shift)
    if shift > 0:
        augmented[:shift] = 0.0
    elif shift < 0:
        augmented[shift:] = 0.0

    noise_std = rng.uniform(0.001, 0.006)
    augmented += rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)

    return np.clip(augmented, -1.0, 1.0)


def oversample_and_augment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes by augmenting minority-class windows until counts match."""
    rng = np.random.default_rng(seed)
    class_indices = {label: np.where(y_train == label)[0] for label in range(len(LABELS))}
    target_count = max(len(indices) for indices in class_indices.values())

    x_balanced = []
    y_balanced = []

    for label, indices in class_indices.items():
        x_label = x_train[indices]
        y_label = y_train[indices]

        if len(indices) < target_count:
            needed = target_count - len(indices)
            extra = []
            for _ in range(needed):
                base_idx = int(rng.choice(indices))
                extra.append(augment_sample(x_train[base_idx], rng))
            x_label = np.concatenate([x_label, np.stack(extra).astype(np.float32)], axis=0)
            y_label = np.concatenate([y_label, np.full(needed, label, dtype=np.int64)], axis=0)

        x_balanced.append(x_label)
        y_balanced.append(y_label)

    x_out = np.concatenate(x_balanced, axis=0)
    y_out = np.concatenate(y_balanced, axis=0)

    shuffle_indices = rng.permutation(len(y_out))
    return x_out[shuffle_indices], y_out[shuffle_indices]

