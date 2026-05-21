from __future__ import annotations

import numpy as np

from .config import (
    BANDPASS_ENABLED,
    BANDPASS_HIGHPASS_CUTOFF_HZ,
    BANDPASS_LOWPASS_CUTOFF_HZ,
    DIFF_FEATURE_GAIN,
    ENERGY_FEATURE_GAIN,
    FEATURE_FRAME_COUNT,
    PREEMPHASIS_ALPHA,
    PREEMPHASIS_ENABLED,
    REMOVE_DC_OFFSET,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    ZCR_FEATURE_GAIN,
)

_TWO_PI = np.float32(2.0 * np.pi)


def _apply_preemphasis(samples: np.ndarray) -> np.ndarray:
    if not PREEMPHASIS_ENABLED or len(samples) == 0:
        return samples
    emphasized = samples.copy()
    emphasized[1:] = samples[1:] - (PREEMPHASIS_ALPHA * samples[:-1])
    return emphasized


def _one_pole_highpass(samples: np.ndarray, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0 or len(samples) == 0:
        return samples

    dt = np.float32(1.0 / SAMPLE_RATE)
    rc = np.float32(1.0 / (_TWO_PI * np.float32(cutoff_hz)))
    alpha = rc / (rc + dt)

    out = np.empty_like(samples)
    previous_input = samples[0]
    previous_output = np.float32(0.0)
    for i, current in enumerate(samples):
        current_output = alpha * (previous_output + current - previous_input)
        out[i] = current_output
        previous_input = current
        previous_output = current_output
    return out


def _one_pole_lowpass(samples: np.ndarray, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0 or len(samples) == 0:
        return samples

    dt = np.float32(1.0 / SAMPLE_RATE)
    rc = np.float32(1.0 / (_TWO_PI * np.float32(cutoff_hz)))
    alpha = dt / (rc + dt)

    out = np.empty_like(samples)
    previous_output = samples[0]
    for i, current in enumerate(samples):
        previous_output = previous_output + alpha * (current - previous_output)
        out[i] = previous_output
    return out


def apply_frontend_dsp(sample: np.ndarray) -> np.ndarray:
    processed = sample.astype(np.float32)
    processed = _apply_preemphasis(processed)
    if BANDPASS_ENABLED:
        processed = _one_pole_highpass(processed, BANDPASS_HIGHPASS_CUTOFF_HZ)
        processed = _one_pole_lowpass(processed, BANDPASS_LOWPASS_CUTOFF_HZ)
    return processed


def extract_voice_features(sample: np.ndarray) -> np.ndarray:
    """Extract ESP32-friendly features with some speaker/timbre signal."""
    window = apply_frontend_dsp(sample)
    if REMOVE_DC_OFFSET:
        window = window - np.mean(window, dtype=np.float32)

    frame_size = WINDOW_SAMPLES // FEATURE_FRAME_COUNT
    trimmed = window[: frame_size * FEATURE_FRAME_COUNT]
    frames = trimmed.reshape(FEATURE_FRAME_COUNT, frame_size)

    energy = np.mean(np.abs(frames), axis=1) * ENERGY_FEATURE_GAIN

    frame_diff = np.diff(frames, axis=1)
    diff_energy = np.mean(np.abs(frame_diff), axis=1) * DIFF_FEATURE_GAIN

    signs = frames >= 0.0
    sign_changes = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    zcr = (sign_changes / max(1, frame_size - 1)) * ZCR_FEATURE_GAIN

    features = np.concatenate([energy, diff_energy, zcr])
    return np.clip(features, 0.0, 1.0).astype(np.float32)


def extract_feature_batch(samples: np.ndarray) -> np.ndarray:
    return np.stack([extract_voice_features(sample) for sample in samples]).astype(np.float32)
