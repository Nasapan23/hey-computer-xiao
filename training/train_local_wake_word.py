from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy import signal
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


LABELS = ["wake_word", "not_wake"]
LABEL_SOURCE_DIRS = {
    "wake_word": ["wake_word"],
    "not_wake": ["unknown_speech", "background_noise"],
}
SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
STRIDE_SECONDS = 0.25
STRIDE_SAMPLES = int(SAMPLE_RATE * STRIDE_SECONDS)
FEATURE_COUNT = 64
FEATURE_GAIN = 200.0
TARGET_RMS = 0.04
MIN_GAIN = 0.5
MAX_GAIN = 20.0
REMOVE_DC_OFFSET = True


def load_wav(path: Path, sample_rate: int) -> np.ndarray:
    original_rate, audio = wavfile.read(path)

    original_dtype = audio.dtype
    is_integer_pcm = np.issubdtype(original_dtype, np.integer)
    max_value = float(np.iinfo(original_dtype).max) if is_integer_pcm else 1.0

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if is_integer_pcm:
        audio = audio.astype(np.float32) / float(max_value)
    else:
        audio = audio.astype(np.float32)

    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        audio = signal.resample_poly(
            audio,
            sample_rate // divisor,
            original_rate // divisor,
        ).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return audio


def slice_windows(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < WINDOW_SAMPLES:
        padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded[None, :], np.array([0], dtype=np.int64)

    windows = []
    starts = []
    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        windows.append(audio[start : start + WINDOW_SAMPLES])
        starts.append(start)

    if not windows:
        windows.append(audio[:WINDOW_SAMPLES])
        starts.append(0)

    return np.stack(windows).astype(np.float32), np.array(starts, dtype=np.int64)


def choose_windows(
    windows: np.ndarray,
    starts: np.ndarray,
    source_label: str,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows) <= max_windows:
        return windows, starts

    if source_label == "background_noise":
        indices = np.linspace(0, len(windows) - 1, max_windows).round().astype(int)
        return windows[indices], starts[indices]

    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    indices = np.argsort(rms)[-max_windows:]
    indices.sort()
    return windows[indices], starts[indices]


def preprocess_windows(windows: np.ndarray, apply_rms_normalization: bool) -> np.ndarray:
    out = windows.copy()
    for i in range(len(out)):
        window = out[i]
        if REMOVE_DC_OFFSET:
            window = window - np.mean(window, dtype=np.float32)

        if apply_rms_normalization:
            rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float32)))
            if rms > 1e-8:
                gain = np.clip(TARGET_RMS / rms, MIN_GAIN, MAX_GAIN)
                window = window * gain

        out[i] = np.clip(window, -1.0, 1.0)
    return out


def write_preview_windows(
    processed_dir: Path,
    wav_path: Path,
    label: str,
    windows: np.ndarray,
    starts: np.ndarray,
) -> None:
    output_dir = processed_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in wav_path.stem)
    for index, (window, start) in enumerate(zip(windows, starts), start=1):
        start_ms = int(round(start * 1000 / SAMPLE_RATE))
        output_path = output_dir / f"{safe_stem}_window_{index:02d}_{start_ms}ms.wav"
        int16_audio = np.clip(window * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(output_path, SAMPLE_RATE, int16_audio)


def make_dataset(
    dataset_dir: Path,
    max_windows_per_file: int,
    apply_rms_normalization: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    examples = []
    targets = []
    sources = []
    manifest = []

    for label_index, label in enumerate(LABELS):
        wav_paths: list[tuple[str, Path]] = []

        for source_label in LABEL_SOURCE_DIRS[label]:
            source_dir = dataset_dir / "raw" / source_label
            source_paths = sorted(
                path for path in source_dir.rglob("*.wav")
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
    # Split by source file, not by individual windows, to avoid train/test leakage.
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    for label_index in range(len(LABELS)):
        label_indices = np.where(y == label_index)[0]
        label_sources = np.unique(source_ids[label_indices])
        rng.shuffle(label_sources)

        test_file_count = max(1, int(round(len(label_sources) * 0.2)))
        test_sources = set(label_sources[:test_file_count])

        for index in label_indices:
            if source_ids[index] in test_sources:
                test_mask[index] = True
            else:
                train_mask[index] = True

    return x[train_mask], y[train_mask], x[test_mask], y[test_mask]


def augment_sample(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = sample.copy()

    # Random gain.
    gain = rng.uniform(0.8, 1.2)
    augmented *= gain

    # Small temporal shift, max +/-100 ms.
    max_shift = int(0.1 * SAMPLE_RATE)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    augmented = np.roll(augmented, shift)
    if shift > 0:
        augmented[:shift] = 0.0
    elif shift < 0:
        augmented[shift:] = 0.0

    # Add low-amplitude white noise.
    noise_std = rng.uniform(0.001, 0.006)
    augmented += rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)

    return np.clip(augmented, -1.0, 1.0)


def extract_energy_features(sample: np.ndarray) -> np.ndarray:
    window = sample.astype(np.float32)
    if REMOVE_DC_OFFSET:
        window = window - np.mean(window, dtype=np.float32)

    frame_size = WINDOW_SAMPLES // FEATURE_COUNT
    trimmed = window[: frame_size * FEATURE_COUNT]
    frames = trimmed.reshape(FEATURE_COUNT, frame_size)
    features = np.mean(np.abs(frames), axis=1) * FEATURE_GAIN
    return np.clip(features, 0.0, 1.0).astype(np.float32)


def extract_feature_batch(samples: np.ndarray) -> np.ndarray:
    return np.stack([extract_energy_features(sample) for sample in samples]).astype(np.float32)


def oversample_and_augment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_COUNT,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_tflite(model: tf.keras.Model, representative_x: np.ndarray, output_path: Path) -> bytes:
    def representative_dataset():
        limit = min(len(representative_x), 100)
        for sample in representative_x[:limit]:
            yield [sample.astype(np.float32)[None, :]]

    input_signature = tf.TensorSpec(
        shape=[1, FEATURE_COUNT],
        dtype=tf.float32,
        name="energy_features",
    )
    concrete_function = tf.function(lambda energy_features: model(energy_features)).get_concrete_function(input_signature)
    frozen_function = convert_variables_to_constants_v2(concrete_function)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_function])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return tflite_model


def c_array_bytes(data: bytes) -> str:
    values = [f"0x{byte:02x}" for byte in data]
    lines = []
    for start in range(0, len(values), 12):
        lines.append("  " + ", ".join(values[start : start + 12]))
    return ",\n".join(lines)


def write_model_header(tflite_model: bytes, header_path: Path) -> None:
    header_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ", ".join(f'"{label}"' for label in LABELS)
    data = c_array_bytes(tflite_model)

    header = f"""#pragma once
#include <cstdint>

constexpr int kWakeWordSampleRate = {SAMPLE_RATE};
constexpr int kWakeWordInputSamples = {WINDOW_SAMPLES};
constexpr int kWakeWordFeatureCount = {FEATURE_COUNT};
constexpr float kWakeWordFeatureGain = {FEATURE_GAIN}f;
constexpr int kWakeWordLabelCount = {len(LABELS)};
constexpr const char *kWakeWordLabels[kWakeWordLabelCount] = {{{labels}}};
constexpr int kWakeWordIndex = 0;

alignas(16) const unsigned char wake_word_model_tflite[] = {{
{data}
}};

const unsigned int wake_word_model_tflite_len = {len(tflite_model)};
"""
    header_path.write_text(header, encoding="utf-8")


def save_training_report(
    output_dir: Path,
    manifest: list[dict],
    history: tf.keras.callbacks.History,
    evaluation: list[float],
    train_count: int,
    test_count: int,
    apply_rms_normalization: bool,
) -> None:
    report = {
        "labels": LABELS,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "feature_count": FEATURE_COUNT,
        "feature_gain": FEATURE_GAIN,
        "remove_dc_offset": REMOVE_DC_OFFSET,
        "apply_rms_normalization": apply_rms_normalization,
        "train_examples": train_count,
        "test_examples": test_count,
        "test_loss": float(evaluation[0]),
        "test_accuracy": float(evaluation[1]),
        "epochs_ran": len(history.history["loss"]),
        "files": manifest,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local TinyML wake-word model from WAV files.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("training/output"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows-per-file", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--normalize-rms", action="store_true", help="Apply per-window RMS normalization (usually off for tiny XIAO-only datasets).")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    processed_dir = args.dataset / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WAV files and selecting 2 second windows from your longer recordings...")
    x, y, source_ids, manifest = make_dataset(args.dataset, args.max_windows_per_file, args.normalize_rms)
    x_train, y_train, x_test, y_test = split_dataset(x, y, source_ids, args.seed)
    x_train, y_train = oversample_and_augment(x_train, y_train, args.seed)

    x_train_model = extract_feature_batch(x_train)
    x_test_model = extract_feature_batch(x_test)

    print(f"Examples: train={len(x_train_model)} test={len(x_test_model)}")
    for label_index, label in enumerate(LABELS):
        print(f"  {label}: {int(np.sum(y == label_index))} windows")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train_model,
        y_train,
        validation_data=(x_test_model, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    evaluation = model.evaluate(x_test_model, y_test, verbose=0)
    print(f"Test accuracy: {evaluation[1]:.3f}")

    keras_path = args.output / "wake_word_model.keras"
    tflite_path = args.output / "wake_word_model_int8.tflite"
    header_path = Path("esp32_tinyml_wake_word") / "model_data.h"

    model.save(keras_path)
    tflite_model = export_tflite(model, x_train_model, tflite_path)
    write_model_header(tflite_model, header_path)
    save_training_report(args.output, manifest, history, evaluation, len(x_train_model), len(x_test_model), args.normalize_rms)

    print()
    print(f"Saved Keras model: {keras_path}")
    print(f"Saved int8 TFLite model: {tflite_path}")
    print(f"Generated Arduino model header: {header_path}")
    print("Next: open esp32_tinyml_wake_word/esp32_tinyml_wake_word.ino and flash it.")


if __name__ == "__main__":
    main()
