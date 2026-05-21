# Local Training Commands

This project trains locally with TensorFlow. Edge Impulse is not required.

The current model has three classes:

- `authorized_user_wake`
- `unknown_user_wake`
- `not_wake`

## Install and Train

From the repo root:

```powershell
python -m venv training\.venv
training\.venv\Scripts\activate
pip install -r training\requirements.txt
python training\train_local_wake_word.py
```

Or double-click:

```text
training/train_local.bat
```

## Your Long Audio Files

Your WAV files can be 3-10 seconds long.

The trainer uses 2 second windows with 0.25 second stride. It does not permanently cut your original files. It selects training chunks and writes copies to:

```text
dataset/processed/
```

Listen to those selected chunks to confirm the wake-word chunks contain the full phrase.

## Outputs

```text
training/output/wake_word_model.keras
training/output/wake_word_model_int8.tflite
training/output/training_report.json
training/output/confusion_matrix.csv
training/output/model_summary.txt
esp32_tinyml_wake_word/model_data.h
```

`training_report.json` includes model layer details and confusion-matrix metrics.

Flash:

```text
esp32_tinyml_wake_word/esp32_tinyml_wake_word.ino
```

## Useful Options

Use more windows from each source recording:

```powershell
python training\train_local_wake_word.py --max-windows-per-file 8
```

Train for more epochs:

```powershell
python training\train_local_wake_word.py --epochs 80
```

Use both:

```powershell
python training\train_local_wake_word.py --epochs 80 --max-windows-per-file 8
```

If accuracy is around `0.500`, check `dataset/processed/` before changing code. Most bad first results come from selected chunks that do not contain the expected audio.
