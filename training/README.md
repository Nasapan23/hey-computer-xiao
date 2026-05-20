# Local TinyML Training

This trains the wake-word model on your PC. It does not use Edge Impulse.

Your WAV files can be 3 to 10 seconds long. The training script automatically searches each file with overlapping 2 second windows, because a TinyML model needs one fixed input size.

## Input Dataset

Put WAV files here:

```text
dataset/raw/wake_word/
dataset/raw/unknown_speech/
dataset/raw/background_noise/
```

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

## What the Script Creates

```text
training/output/wake_word_model.keras
training/output/wake_word_model_int8.tflite
training/output/training_report.json
esp32_tinyml_wake_word/model_data.h
```

`model_data.h` is the TinyML model converted into a C array for Arduino.

## How 3-10 Second Audio Is Used

The ESP32 model input is 2 seconds of audio:

```text
32000 samples at 16000 Hz
```

For each longer WAV file, the script creates overlapping 2 second windows every 0.25 seconds.

For `wake_word` and `unknown_speech`, it keeps the loudest windows from each file. This helps when your file contains silence before or after the spoken phrase.

For `background_noise`, it keeps windows spread across the file.

The selected windows are saved here so you can listen to them:

```text
dataset/processed/
```

## Training Stability Improvements

The trainer now uses:

- file-level train/test split: windows from the same original file never appear in both train and test
- class balancing with augmentation: minority classes are oversampled using small gain changes, small time shifts, and light noise
- cleaned processed previews: `dataset/processed/` is regenerated on each run

## Important Dataset Tip

If a `wake_word` WAV is 10 seconds long but contains the wake word only once, the script will try to pick the loud 2 second area that contains the phrase. This is okay for a first experiment.

Still, the best training files are not huge recordings. Best first version:

- `wake_word`: each file should contain one clear "hey computer", with some silence before or after
- `unknown_speech`: each file should contain other spoken words
- `background_noise`: each file should contain no speech

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

1. Open `dataset/processed/wake_word/`.
2. Listen to several selected chunks.
3. Confirm each one contains the full wake phrase.
4. Open `dataset/processed/unknown_speech/`.
5. Confirm those do not contain the wake phrase.
6. Open `dataset/processed/background_noise/`.
7. Confirm those contain no speech.

Then improve the data:

- Add more `wake_word` recordings.
- Add more `unknown_speech` recordings that sound similar but are not the wake phrase.
- Add more background recordings from the same room.
- Retrain.

For a better first model, aim for at least 30 source WAV files per label.
