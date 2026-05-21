from __future__ import annotations

import numpy as np

from .config import LABELS


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def compute_per_class_metrics(confusion_matrix: np.ndarray) -> list[dict]:
    total = int(np.sum(confusion_matrix))
    metrics = []

    for label_index, label in enumerate(LABELS):
        tp = int(confusion_matrix[label_index, label_index])
        fn = int(np.sum(confusion_matrix[label_index, :]) - tp)
        fp = int(np.sum(confusion_matrix[:, label_index]) - tp)
        tn = total - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append(
            {
                "label": label,
                "support": int(np.sum(confusion_matrix[label_index, :])),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    return metrics


def print_confusion_matrix(confusion_matrix: np.ndarray) -> None:
    print("Confusion matrix (rows=true, cols=pred):")
    header = " " * 18 + " ".join(f"{label:>12}" for label in LABELS)
    print(header)
    for row_index, label in enumerate(LABELS):
        row_values = " ".join(f"{int(value):>12}" for value in confusion_matrix[row_index])
        print(f"{label:>18} {row_values}")

