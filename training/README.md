# Local TinyML Training

This trains the wake-word model on your PC. It does not use Edge Impulse.

Your WAV files can be 3 to 10 seconds long. The training script automatically searches each file with overlapping 2 second windows, because a TinyML model needs one fixed input size.

## Input Dataset

Put WAV files here:

```text
dataset/raw/authorized_user_wake/
dataset/raw/unknown_user_wake/
dataset/raw/unknown_speech/
```

Model classes:

- `authorized_user_wake`: your voice saying "hey computer"
- `unknown_user_wake`: someone else saying "hey computer"
- `not_wake`: from `unknown_speech`

Legacy folders `dataset/raw/wake_word/` and `dataset/raw/background_noise/` are now ignored by training.

The script accepts common WAV sample rates and resamples to `16000 Hz`.

## Run Training

On Windows, double-click:

```text
training/train_local.bat
```

Or run manually:

```powershell
python -m venv training\.venv
training\.venv\Scripts\activate
pip install -r training\requirements.txt
python training\train_local_wake_word.py
```

Default training now runs for `100` epochs (with early stopping enabled).  
To override:

```powershell
python training\train_local_wake_word.py --epochs 40
```

## Code Layout

```text
training/
|-- train_local_wake_word.py   command-line entry point
|-- train_local.bat            Windows launcher
|-- requirements.txt
`-- wake_word/
    |-- config.py              labels and audio/model constants
    |-- audio.py               WAV loading, windowing, preview writing
    |-- dataset.py             dataset manifest and train/test split
    |-- augmentation.py        balancing and light audio augmentation
    |-- features.py            ESP32-matched voice features
    |-- model.py               Keras model architecture
    |-- metrics.py             confusion matrix and class metrics
    |-- export.py              TFLite + model_data.h export
    |-- reports.py             JSON/CSV/text training reports
    `-- pipeline.py            end-to-end training workflow
```

## What the Script Creates

```text
training/output/wake_word_model.keras
training/output/wake_word_model_int8.tflite
training/output/training_report.json
training/output/confusion_matrix.csv
training/output/confusion_matrix.png
training/output/training_curves.png
training/output/class_distribution.png
training/output/per_class_metrics.png
training/output/model_summary.txt
esp32_tinyml_wake_word/model_data.h
```

`model_data.h` is the TinyML model converted into a C array for Arduino.

## How 3-10 Second Audio Is Used

The ESP32 model input is 2 seconds of audio:

```text
32000 samples at 16000 Hz
```

For each longer WAV file, the script creates overlapping 2 second windows every 0.25 seconds.

For `authorized_user_wake`, `unknown_user_wake`, and `unknown_speech`, it keeps the loudest windows from each file. This helps when your file contains silence before or after the spoken phrase.

The selected windows are saved here so you can listen to them:

```text
dataset/processed/
```

## Training Stability Improvements

The trainer now uses:

- file-level train/test split: windows from the same original file never appear in both train and test
- class balancing with augmentation: minority classes are oversampled using small gain changes, small time shifts, and light noise
- cleaned processed previews: `dataset/processed/` is regenerated on each run

`training_report.json` also records:

- epoch-by-epoch training/validation metrics
- full model layer details (params, shapes, activations)
- confusion matrix and per-class metrics

And the trainer now exports visual artifacts in `training/output/`:

- `confusion_matrix.png`
- `training_curves.png`
- `class_distribution.png`
- `per_class_metrics.png`

## Important Dataset Tip

If a wake WAV is 10 seconds long but contains the wake word only once, the script will try to pick the loud 2 second area that contains the phrase. This is okay for a first experiment.

Still, the best training files are not huge recordings. Best first version:

- `authorized_user_wake`: each file should contain your clear "hey computer", with some silence before or after
- `unknown_user_wake`: each file should contain another person saying "hey computer"
- `unknown_speech`: each file should contain other spoken words and non-wake audio (silence/noise is fine too)

## After Training

Flash:

```text
esp32_tinyml_wake_word/esp32_tinyml_wake_word.ino
```

That sketch includes:

```cpp
#include "model_data.h"
```

So after each retrain, flash the sketch again.

## About Accuracy

`Test accuracy: 0.500` means the model is only right about half the time on the small test split. That is not good enough for wake-word use.

With your current dataset size, accuracy can jump around a lot because there are only a few test examples. Use it as a warning, not as a final scientific score.

First checks:

1. Open `dataset/processed/authorized_user_wake/`.
2. Confirm those chunks contain you saying the full wake phrase.
3. Open `dataset/processed/unknown_user_wake/`.
4. Confirm those chunks contain someone else saying the full wake phrase.
5. Open `dataset/processed/unknown_speech/`.
6. Confirm those do not contain the wake phrase.

Then improve the data:

- Add more `authorized_user_wake` recordings from you.
- Add more `unknown_user_wake` recordings from other people.
- Add more `unknown_speech` recordings that sound similar but are not the wake phrase.
- Retrain.

For a better first model, aim for at least 30 source WAV files per label.
