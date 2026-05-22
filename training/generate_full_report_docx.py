from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docx import Document
from docx.shared import Inches


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_round(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def add_heading_paragraph(document: Document, text: str, level: int = 1) -> None:
    document.add_heading(text, level=level)


def add_kv_table(document: Document, rows: list[tuple[str, str]]) -> None:
    table = document.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    header = table.rows[0].cells
    header[0].text = "Field"
    header[1].text = "Value"
    for key, value in rows:
        row = table.add_row().cells
        row[0].text = key
        row[1].text = value


def add_list(document: Document, lines: list[str]) -> None:
    for line in lines:
        document.add_paragraph(line, style="List Bullet")


def parse_runtime_constants(ino_path: Path) -> dict[str, str]:
    constants: dict[str, str] = {}
    pattern = re.compile(r"^\s*constexpr\s+\w+\s+([A-Z0-9_]+)\s*=\s*([^;]+);")
    for line in ino_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.groups()
        constants[key] = value.strip()
    return constants


def collect_transcription_stats(commands_root: Path) -> tuple[Counter[str], list[dict[str, str]]]:
    status_counts: Counter[str] = Counter()
    records: list[dict[str, str]] = []

    for sidecar_path in sorted(commands_root.rglob("*.transcript.json")):
        try:
            payload = read_json(sidecar_path)
        except Exception:
            status_counts["parse_error"] += 1
            continue
        status = str(payload.get("transcription_status", "")).strip() or "missing"
        transcript = str(payload.get("transcript", "")).strip()
        language = str(payload.get("transcription_language", "")).strip()
        error = str(payload.get("transcription_error", "")).strip()
        status_counts[status] += 1
        records.append(
            {
                "file": sidecar_path.name,
                "status": status,
                "language": language or "n/a",
                "transcript": transcript,
                "error": error,
            }
        )
    records.reverse()
    return status_counts, records


def count_raw_wavs(dataset_raw: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label_dir in sorted(dataset_raw.iterdir()):
        if label_dir.is_dir():
            counts[label_dir.name] = len(list(label_dir.glob("*.wav")))
    return counts


def add_image_if_exists(document: Document, image_path: Path, width: float = 6.5) -> None:
    if not image_path.exists():
        document.add_paragraph(f"Missing image: {image_path}")
        return
    document.add_picture(str(image_path), width=Inches(width))


def read_confusion_csv(csv_path: Path) -> list[list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def add_confusion_table(document: Document, confusion_rows: list[list[str]]) -> None:
    if not confusion_rows:
        document.add_paragraph("No confusion matrix CSV rows found.")
        return

    cols = len(confusion_rows[0])
    table = document.add_table(rows=1, cols=cols)
    table.style = "Light Grid Accent 2"
    for i, col_name in enumerate(confusion_rows[0]):
        table.rows[0].cells[i].text = col_name or "label"

    for row_values in confusion_rows[1:]:
        row = table.add_row().cells
        for i, value in enumerate(row_values):
            row[i].text = value


def add_model_summary_block(document: Document, model_summary_path: Path) -> None:
    if not model_summary_path.exists():
        document.add_paragraph("Model summary file missing.")
        return
    text = model_summary_path.read_text(encoding="utf-8")
    document.add_paragraph(text)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    training_output = repo_root / "training" / "output"
    report_path = training_output / "training_report.json"
    model_summary_path = training_output / "model_summary.txt"
    confusion_csv_path = training_output / "confusion_matrix.csv"
    commands_root = repo_root / "server" / "recordings" / "commands"
    dataset_raw = repo_root / "dataset" / "raw"
    ino_path = repo_root / "esp32_tinyml_wake_word" / "esp32_tinyml_wake_word.ino"

    training_report = read_json(report_path)
    runtime_constants = parse_runtime_constants(ino_path)
    status_counts, transcript_records = collect_transcription_stats(commands_root)
    raw_counts = count_raw_wavs(dataset_raw)
    confusion_rows = read_confusion_csv(confusion_csv_path)

    now_utc = datetime.now(timezone.utc)
    stamp = now_utc.strftime("%Y%m%d_%H%M%S")
    output_docx = training_output / f"wake_word_full_report_{stamp}.docx"

    document = Document()
    document.add_heading("Wake Word + Voice Authorization Full Report", level=0)
    document.add_paragraph(f"Generated at UTC: {now_utc.isoformat()}")
    document.add_paragraph(f"Source training report timestamp: {training_report.get('generated_at_utc', 'n/a')}")

    add_heading_paragraph(document, "1. Executive Summary", level=1)
    add_kv_table(
        document,
        [
            ("Test Accuracy", safe_round(training_report.get("test_accuracy"))),
            ("Test Loss", safe_round(training_report.get("test_loss"))),
            ("Epochs Ran", str(training_report.get("epochs_ran", "n/a"))),
            ("Train Examples", str(training_report.get("train_examples", "n/a"))),
            ("Test Examples", str(training_report.get("test_examples", "n/a"))),
            ("Command Transcript Sidecars", str(sum(status_counts.values()))),
            ("Ready Transcripts", str(status_counts.get("ready", 0))),
            ("No Speech", str(status_counts.get("no_speech", 0))),
            ("Unavailable", str(status_counts.get("unavailable", 0))),
        ],
    )

    document.add_paragraph("Current run highlights:")
    add_list(
        document,
        [
            "The classifier uses 3 labels: authorized_user_wake, unknown_user_wake, not_wake.",
            "Unknown-speech clips are mapped to not_wake; legacy wake_word/background_noise folders are excluded from training.",
            "On-paper metrics improved, but runtime triggering remains sensitive to microphone level and decision thresholds.",
        ],
    )

    add_heading_paragraph(document, "2. Dataset Overview", level=1)
    label_summary = training_report.get("label_count_summary", {})
    add_kv_table(
        document,
        [
            ("all_windows", str(label_summary.get("all_windows", "n/a"))),
            ("train_before_augmentation", str(label_summary.get("train_before_augmentation", "n/a"))),
            ("test_split", str(label_summary.get("test_split", "n/a"))),
            ("train_after_augmentation", str(label_summary.get("train_after_augmentation", "n/a"))),
        ],
    )

    document.add_paragraph("Raw WAV files per dataset folder:")
    raw_table = document.add_table(rows=1, cols=2)
    raw_table.style = "Light Grid Accent 1"
    raw_table.rows[0].cells[0].text = "Folder"
    raw_table.rows[0].cells[1].text = "WAV Count"
    for folder, count in raw_counts.items():
        row = raw_table.add_row().cells
        row[0].text = folder
        row[1].text = str(count)

    add_heading_paragraph(document, "3. Training Configuration", level=1)
    setup = training_report.get("training_setup", {})
    add_kv_table(
        document,
        [
            ("Hidden Units", str(setup.get("model_hidden_units", "n/a"))),
            ("Dropout Rates", str(setup.get("model_dropout_rates", "n/a"))),
            ("Gaussian Noise Stddev", str(setup.get("model_gaussian_noise_stddev", "n/a"))),
            ("L2 Regularization", str(setup.get("model_l2_regularization", "n/a"))),
            ("Learning Rate", str(setup.get("model_learning_rate", "n/a"))),
            ("Early Stopping Patience", str(setup.get("early_stopping_patience", "n/a"))),
            ("LR Plateau Patience", str(setup.get("lr_plateau_patience", "n/a"))),
            ("LR Plateau Factor", str(setup.get("lr_plateau_factor", "n/a"))),
            ("Min Learning Rate", str(setup.get("min_learning_rate", "n/a"))),
            ("Class Weights", str(setup.get("class_weight", "n/a"))),
        ],
    )

    add_heading_paragraph(document, "4. Model Information", level=1)
    model_info = training_report.get("model", {})
    add_kv_table(
        document,
        [
            ("Total Params", str(model_info.get("total_params", "n/a"))),
            ("Trainable Params", str(model_info.get("trainable_params", "n/a"))),
            ("Non-Trainable Params", str(model_info.get("non_trainable_params", "n/a"))),
            ("Model Summary Path", str(model_info.get("summary_path", "n/a"))),
            ("Keras Model", str((training_output / "wake_word_model.keras").resolve())),
            ("INT8 TFLite Model", str((training_output / "wake_word_model_int8.tflite").resolve())),
        ],
    )
    document.add_paragraph("Model summary (text export):")
    add_model_summary_block(document, model_summary_path)

    add_heading_paragraph(document, "5. Evaluation Metrics", level=1)
    per_class = training_report.get("per_class_metrics", [])
    metric_table = document.add_table(rows=1, cols=6)
    metric_table.style = "Light Grid Accent 2"
    headers = metric_table.rows[0].cells
    headers[0].text = "Label"
    headers[1].text = "Support"
    headers[2].text = "Precision"
    headers[3].text = "Recall"
    headers[4].text = "F1"
    headers[5].text = "TP / FP / FN / TN"
    for item in per_class:
        row = metric_table.add_row().cells
        row[0].text = str(item.get("label", "n/a"))
        row[1].text = str(item.get("support", "n/a"))
        row[2].text = safe_round(item.get("precision"))
        row[3].text = safe_round(item.get("recall"))
        row[4].text = safe_round(item.get("f1_score"))
        row[5].text = (
            f"{item.get('tp', 'n/a')} / {item.get('fp', 'n/a')} / "
            f"{item.get('fn', 'n/a')} / {item.get('tn', 'n/a')}"
        )

    document.add_paragraph("Confusion matrix table (rows=true, cols=pred):")
    add_confusion_table(document, confusion_rows)

    add_heading_paragraph(document, "6. Charts", level=1)
    for title, img_name in [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("Training Curves", "training_curves.png"),
        ("Per-Class Metrics", "per_class_metrics.png"),
        ("Class Distribution", "class_distribution.png"),
    ]:
        document.add_heading(title, level=2)
        add_image_if_exists(document, training_output / img_name)

    add_heading_paragraph(document, "7. Runtime Detector Configuration", level=1)
    keys_to_report = [
        "DEVICE_MODE",
        "AUTHORIZED_WAKE_THRESHOLD",
        "UNKNOWN_USER_WAKE_THRESHOLD",
        "AUTHORIZED_WAKE_MARGIN",
        "WAKE_DEBOUNCE_MS",
        "UNKNOWN_WAKE_DEBOUNCE_MS",
        "SPEECH_GATE_ENABLED",
        "SPEECH_GATE_MIN_CENTERED_RMS",
        "SPEECH_GATE_MIN_P2P",
        "SPEECH_GATE_REQUIRE_BOTH_SIGNALS",
        "SPEECH_GATE_HANGOVER_MS",
        "SCORE_SMOOTHING_ENABLED",
        "SCORE_SMOOTHING_ALPHA",
    ]
    runtime_rows: list[tuple[str, str]] = []
    for key in keys_to_report:
        runtime_rows.append((key, runtime_constants.get(key, "n/a")))
    add_kv_table(document, runtime_rows)

    add_heading_paragraph(document, "8. Command Transcription Analysis", level=1)
    add_kv_table(
        document,
        [
            ("Total Sidecars", str(sum(status_counts.values()))),
            ("ready", str(status_counts.get("ready", 0))),
            ("no_speech", str(status_counts.get("no_speech", 0))),
            ("unavailable", str(status_counts.get("unavailable", 0))),
            ("error", str(status_counts.get("error", 0))),
            ("missing", str(status_counts.get("missing", 0))),
            ("parse_error", str(status_counts.get("parse_error", 0))),
        ],
    )

    document.add_paragraph("Recent transcription samples:")
    sample_table = document.add_table(rows=1, cols=4)
    sample_table.style = "Light Grid Accent 1"
    sample_table.rows[0].cells[0].text = "File"
    sample_table.rows[0].cells[1].text = "Status"
    sample_table.rows[0].cells[2].text = "Language"
    sample_table.rows[0].cells[3].text = "Transcript / Error (trimmed)"
    for item in transcript_records[-12:]:
        row = sample_table.add_row().cells
        row[0].text = item["file"]
        row[1].text = item["status"]
        row[2].text = item["language"]
        text = item["transcript"] if item["transcript"] else item["error"]
        row[3].text = (text[:200] + "...") if len(text) > 200 else text

    add_heading_paragraph(document, "9. Findings and Next Actions", level=1)
    add_list(
        document,
        [
            "The latest model is usable but still confuses authorized and unknown-user wake classes under weak or distant speech.",
            "Runtime thresholds now prioritize detection for demos; this increases catch rate but may increase false accepts.",
            "To improve authorized recall, capture more authorized clips with varied distance, speed, and microphone angle.",
            "To reduce confusion, record hard-negative unknown_user_wake clips spoken in similar rhythm to the authorized phrase.",
            "For production reliability, tune thresholds separately from training and validate using live-stream logs.",
        ],
    )

    add_heading_paragraph(document, "10. Artifact Paths", level=1)
    artifact_table = document.add_table(rows=1, cols=2)
    artifact_table.style = "Light Grid Accent 2"
    artifact_table.rows[0].cells[0].text = "Artifact"
    artifact_table.rows[0].cells[1].text = "Path"
    artifact_paths = [
        ("Training report JSON", report_path),
        ("Confusion matrix CSV", confusion_csv_path),
        ("Model summary", model_summary_path),
        ("Keras model", training_output / "wake_word_model.keras"),
        ("INT8 TFLite model", training_output / "wake_word_model_int8.tflite"),
        ("Confusion matrix image", training_output / "confusion_matrix.png"),
        ("Training curves image", training_output / "training_curves.png"),
        ("Per-class metrics image", training_output / "per_class_metrics.png"),
        ("Class distribution image", training_output / "class_distribution.png"),
        ("Commands transcription root", commands_root),
        ("ESP32 firmware sketch", ino_path),
    ]
    for name, path in artifact_paths:
        row = artifact_table.add_row().cells
        row[0].text = name
        row[1].text = str(path.resolve())

    document.save(str(output_docx))
    print(output_docx)


if __name__ == "__main__":
    main()
