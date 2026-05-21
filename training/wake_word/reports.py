from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    BANDPASS_ENABLED,
    BANDPASS_HIGHPASS_CUTOFF_HZ,
    BANDPASS_LOWPASS_CUTOFF_HZ,
    DIFF_FEATURE_GAIN,
    ENERGY_FEATURE_GAIN,
    FEATURE_COUNT,
    FEATURE_FRAME_COUNT,
    FEATURE_GROUP_COUNT,
    FEATURE_LAYOUT,
    LABELS,
    PREEMPHASIS_ALPHA,
    PREEMPHASIS_ENABLED,
    REMOVE_DC_OFFSET,
    SAMPLE_RATE,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
    ZCR_FEATURE_GAIN,
)


def pretty_label(label: str) -> str:
    return label.replace("_", " ")


def describe_model_layers(model: tf.keras.Model) -> list[dict]:
    layers = []
    for layer in model.layers:
        layer_config = layer.get_config()
        try:
            output_shape = tf.TensorShape(layer.output.shape).as_list()
        except Exception:
            output_shape = None

        layers.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": output_shape,
                "params": int(layer.count_params()),
                "units": layer_config.get("units"),
                "activation": layer_config.get("activation"),
            }
        )

    return layers


def write_confusion_matrix_csv(output_dir: Path, confusion_matrix: np.ndarray) -> Path:
    csv_path = output_dir / "confusion_matrix.csv"
    lines = ["," + ",".join(f"pred_{label}" for label in LABELS)]
    for row_index, label in enumerate(LABELS):
        row_values = ",".join(str(int(value)) for value in confusion_matrix[row_index])
        lines.append(f"true_{label},{row_values}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def write_model_summary_text(output_dir: Path, model: tf.keras.Model) -> Path:
    summary_path = output_dir / "model_summary.txt"
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return summary_path


def write_confusion_matrix_image(output_dir: Path, confusion_matrix: np.ndarray) -> Path:
    image_path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [pretty_label(label) for label in LABELS]
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = confusion_matrix.max() / 2.0 if confusion_matrix.size > 0 else 0.0
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[row, col])
            color = "white" if confusion_matrix[row, col] > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    return image_path


def write_training_curves_image(output_dir: Path, training_history: dict[str, list[float]]) -> Path:
    image_path = output_dir / "training_curves.png"

    train_loss = training_history.get("loss", [])
    val_loss = training_history.get("val_loss", [])
    train_accuracy = training_history.get("accuracy", [])
    val_accuracy = training_history.get("val_accuracy", [])
    epoch_count = max(len(train_loss), len(train_accuracy))
    epochs = np.arange(1, epoch_count + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss, ax_acc = axes

    if len(train_loss) > 0:
        ax_loss.plot(epochs[: len(train_loss)], train_loss, label="train_loss", linewidth=2)
    if len(val_loss) > 0:
        ax_loss.plot(epochs[: len(val_loss)], val_loss, label="val_loss", linewidth=2)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best")

    if len(train_accuracy) > 0:
        ax_acc.plot(epochs[: len(train_accuracy)], train_accuracy, label="train_accuracy", linewidth=2)
    if len(val_accuracy) > 0:
        ax_acc.plot(epochs[: len(val_accuracy)], val_accuracy, label="val_accuracy", linewidth=2)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="best")

    fig.tight_layout()
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    return image_path


def write_class_distribution_image(output_dir: Path, label_count_summary: dict[str, list[int]]) -> Path:
    image_path = output_dir / "class_distribution.png"

    split_order = [
        ("all_windows", "all"),
        ("train_before_augmentation", "train_pre_aug"),
        ("test_split", "test"),
        ("train_after_augmentation", "train_post_aug"),
    ]

    series = [(display_name, label_count_summary[key]) for key, display_name in split_order if key in label_count_summary]
    x = np.arange(len(LABELS))
    width = 0.18 if len(series) >= 4 else 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.linspace(-(len(series) - 1) / 2.0, (len(series) - 1) / 2.0, len(series))
    for offset, (name, values) in zip(offsets, series):
        ax.bar(x + (offset * width), values, width=width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(label) for label in LABELS], rotation=15, ha="right")
    ax.set_ylabel("Window count")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    return image_path


def write_per_class_metrics_image(output_dir: Path, per_class_metrics: list[dict]) -> Path:
    image_path = output_dir / "per_class_metrics.png"

    labels = [pretty_label(metric["label"]) for metric in per_class_metrics]
    precision = [float(metric["precision"]) for metric in per_class_metrics]
    recall = [float(metric["recall"]) for metric in per_class_metrics]
    f1 = [float(metric["f1_score"]) for metric in per_class_metrics]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width=width, label="precision")
    ax.bar(x, recall, width=width, label="recall")
    ax.bar(x + width, f1, width=width, label="f1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    return image_path


def save_training_report(
    output_dir: Path,
    manifest: list[dict],
    history: tf.keras.callbacks.History,
    evaluation: list[float],
    confusion_matrix: np.ndarray,
    per_class_metrics: list[dict],
    model: tf.keras.Model,
    train_count: int,
    test_count: int,
    evaluation_uses_training_data: bool,
    apply_rms_normalization: bool,
    label_count_summary: dict[str, list[int]],
) -> None:
    """Persist all training metadata needed for debugging and report writing."""
    training_history = {key: [float(value) for value in values] for key, values in history.history.items()}
    model_layers = describe_model_layers(model)
    confusion_csv_path = write_confusion_matrix_csv(output_dir, confusion_matrix)
    model_summary_path = write_model_summary_text(output_dir, model)
    confusion_matrix_image_path = write_confusion_matrix_image(output_dir, confusion_matrix)
    training_curves_image_path = write_training_curves_image(output_dir, training_history)
    class_distribution_image_path = write_class_distribution_image(output_dir, label_count_summary)
    per_class_metrics_image_path = write_per_class_metrics_image(output_dir, per_class_metrics)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": LABELS,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "feature_frame_count": FEATURE_FRAME_COUNT,
        "feature_group_count": FEATURE_GROUP_COUNT,
        "feature_count": FEATURE_COUNT,
        "feature_layout": FEATURE_LAYOUT,
        "energy_feature_gain": ENERGY_FEATURE_GAIN,
        "diff_feature_gain": DIFF_FEATURE_GAIN,
        "zcr_feature_gain": ZCR_FEATURE_GAIN,
        "remove_dc_offset": REMOVE_DC_OFFSET,
        "preemphasis_enabled": PREEMPHASIS_ENABLED,
        "preemphasis_alpha": PREEMPHASIS_ALPHA,
        "bandpass_enabled": BANDPASS_ENABLED,
        "bandpass_highpass_cutoff_hz": BANDPASS_HIGHPASS_CUTOFF_HZ,
        "bandpass_lowpass_cutoff_hz": BANDPASS_LOWPASS_CUTOFF_HZ,
        "apply_rms_normalization": apply_rms_normalization,
        "label_count_summary": label_count_summary,
        "train_examples": train_count,
        "test_examples": test_count,
        "evaluation_uses_training_data": evaluation_uses_training_data,
        "test_loss": float(evaluation[0]),
        "test_accuracy": float(evaluation[1]),
        "epochs_ran": len(history.history["loss"]),
        "training_history": training_history,
        "model": {
            "total_params": int(model.count_params()),
            "trainable_params": int(np.sum([np.prod(v.shape) for v in model.trainable_weights])),
            "non_trainable_params": int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights])),
            "layers": model_layers,
            "summary_path": str(model_summary_path),
        },
        "confusion_matrix": {
            "labels": LABELS,
            "rows_true_cols_pred": confusion_matrix.tolist(),
            "csv_path": str(confusion_csv_path),
            "image_path": str(confusion_matrix_image_path),
        },
        "per_class_metrics": per_class_metrics,
        "artifacts": {
            "training_curves_image_path": str(training_curves_image_path),
            "class_distribution_image_path": str(class_distribution_image_path),
            "per_class_metrics_image_path": str(per_class_metrics_image_path),
        },
        "files": manifest,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
