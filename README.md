# Local TinyML Wake Word on XIAO ESP32S3 Sense

This project trains and runs a wake-word model locally.

No Edge Impulse is required for the main path.

Flow:

1. Put your audio files in `dataset/raw/...`.
2. Train the model locally with Python and TensorFlow.
3. The trainer exports an int8 TinyML model as `model_data.h`.
4. Flash the XIAO ESP32S3 Sense with the Arduino TinyML sketch.
5. Run the Python PC server.
6. Say "hey computer". If it matches the authorized user's voice, the ESP32 sends a ping, records the next 5 seconds, and uploads that audio to the PC.

The ESP32 does not do speech-to-text. It detects authorized wake, unknown-user wake, or not-wake, then uploads command audio only after an authorized wake.

## Your 3-10 Second Recordings Are OK

Your dataset files do not need to be exactly 1 second.

The local trainer scans each WAV file using overlapping 2 second windows:

```text
0.00s - 2.00s
0.25s - 2.25s
0.50s - 2.50s
...
```

For `authorized_user_wake`, `unknown_user_wake`, and `unknown_speech`, it keeps the loudest windows from each file. This usually keeps the part where someone spoke.

For `background_noise`, it keeps windows spread across the whole file.

The selected chunks are written to:

```text
dataset/processed/
```

Listen to those files after training starts. If a wake chunk does not contain the full "hey computer" phrase, trim or re-record that source file.

## Folder Layout

```text
.
|-- dataset/
|   |-- raw/
|   |   |-- authorized_user_wake/
|   |   |-- unknown_user_wake/
|   |   |-- unknown_speech/
|   |   `-- background_noise/
|   `-- processed/
|-- training/
|   |-- train_local.bat
|   |-- train_local_wake_word.py
|   |-- wake_word/
|   |   |-- config.py
|   |   |-- dataset.py
|   |   |-- features.py
|   |   |-- model.py
|   |   |-- pipeline.py
|   |   `-- reports.py
|   `-- requirements.txt
|-- esp32_tinyml_wake_word/
|   |-- esp32_tinyml_wake_word.ino
|   |-- audio_runtime.h
|   |-- tinyml_runtime.h
|   |-- network_client.h
|   |-- collection_mode.h
|   |-- wake_detection.h
|   `-- model_data.h          created by training
`-- server/
    |-- main.py
    |-- wake_server/
    |   |-- app.py
    |   |-- routes.py
    |   |-- audio_io.py
    |   |-- config.py
    |   `-- state.py
    |-- run_server.bat
    `-- test_upload.py
```

`train_local_wake_word.py`, `server/main.py`, and `esp32_tinyml_wake_word.ino` are short entry points. Implementation details live in smaller files beside them.

## Step 1: Check Dataset Folders

Use exactly these labels:

```text
dataset/raw/authorized_user_wake/
dataset/raw/unknown_user_wake/
dataset/raw/unknown_speech/
dataset/raw/background_noise/
```

The trainer also accepts old files in `dataset/raw/wake_word/` as authorized-user wake clips.

The model outputs three classes:

- `authorized_user_wake`: your voice saying "hey computer"
- `unknown_user_wake`: someone else saying "hey computer"
- `not_wake`: other speech and background noise

Your current files can be 3-10 seconds long.

Best format:

- WAV
- mono or stereo is OK
- 16000 Hz preferred, but the trainer can resample
- `authorized_user_wake` files should contain you saying the wake phrase
- `unknown_user_wake` files should contain other people saying the wake phrase
- `unknown_speech` files should contain speech that is not the wake phrase
- `background_noise` files should contain no speech

## Step 2: Train Locally

On Windows, double-click:

```text
training/train_local.bat
```

Or run manually from the repo root:

```powershell
python -m venv training\.venv
training\.venv\Scripts\activate
pip install -r training\requirements.txt
python training\train_local_wake_word.py
```

Training creates:

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
dataset/processed/
```

`model_data.h` is the TinyML model that gets compiled into the ESP32 firmware.

## Step 3: Run the PC Server

Double-click:

```text
server/run_server.bat
```

Or run manually:

```powershell
cd server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test server health:

```powershell
curl http://127.0.0.1:8000/health
```

Test ping without the ESP32:

```powershell
curl -X POST http://127.0.0.1:8000/ping
```

Check current wake-word count:

```text
http://127.0.0.1:8000/counter
```

Open the live signal + command-audio monitor:

```text
http://127.0.0.1:8000/ui
```

## Step 4: Flash the ESP32

Open:

```text
esp32_tinyml_wake_word/esp32_tinyml_wake_word.ino
```

Edit these values:

```cpp
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char *PC_SERVER_PING_URL = "http://192.168.1.50:8000/ping";
const char *PC_SERVER_UPLOAD_URL = "http://192.168.1.50:8000/upload";
```

Use your PC's real LAN IP address. Do not use `localhost`.

Arduino IDE setup:

1. Install Arduino IDE 2.x.
2. Install the Espressif ESP32 board package.
3. Install `TensorFlowLite_ESP32` or another TensorFlow Lite Micro Arduino port for ESP32.
4. Select XIAO ESP32S3.
5. Enable PSRAM if the board menu has that option.
6. Upload the sketch.
7. Open Serial Monitor at `115200`.

## What the ESP32 Does

The firmware continuously keeps a rolling 2 second audio window from the onboard microphone and advances it every 0.25 seconds.

For every inference:

1. Convert raw microphone samples to the model input.
2. Run TensorFlow Lite Micro.
3. Score `authorized_user_wake`, `unknown_user_wake`, and `not_wake`.
4. If authorized wake wins with enough confidence, send `/ping`.
5. Record the next 5 seconds of audio.
6. Upload that command audio with HTTP POST `/upload`.
7. If unknown-user wake wins, log/reject it without uploading command audio.

## Settings You May Change

In the ESP32 sketch:

```cpp
constexpr float AUTHORIZED_WAKE_THRESHOLD = 0.65f;
constexpr float UNKNOWN_USER_WAKE_THRESHOLD = 0.65f;
constexpr float COMMAND_CAPTURE_SECONDS = 5.0f;
constexpr uint32_t WAKE_DEBOUNCE_MS = 3000;
```

In the trainer:

```python
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.25
```

Use `2.0` seconds for "hey computer". A 1 second window can cut the phrase too much.

`training/output/training_report.json` now includes:

- training history per epoch
- layer-by-layer model details (type, params, output shape, activation)
- confusion matrix values
- per-class precision/recall/F1

## If Training Looks Bad

Check `dataset/processed/`.

If the selected wake chunks do not contain the full phrase:

- trim the original recording closer to the phrase, or
- record shorter wake-word clips, or
- increase `--max-windows-per-file` when running the trainer.

Example:

```powershell
python training\train_local_wake_word.py --max-windows-per-file 8
```

If the ESP32 triggers too easily:

- raise `AUTHORIZED_WAKE_THRESHOLD`
- add more `unknown_user_wake` clips from other people
- add more `unknown_speech`
- add more `background_noise`
- retrain and flash again

## Optional: XIAO Mic Data Collection Mode

You can collect labeled clips directly from the XIAO microphone and save them into your training dataset.

1. Start the PC server (`server/run_server.bat`) so `/collect` is available.
2. Open `esp32_tinyml_wake_word/esp32_tinyml_wake_word.ino` and set:

```cpp
constexpr DeviceMode DEVICE_MODE = MODE_COLLECT_DATA;
```

3. Flash the board and open Serial Monitor at `115200`.
4. Use Serial commands:

- `a` -> capture/upload one `authorized_user_wake` clip from you
- `u` -> capture/upload one `unknown_user_wake` clip from someone else saying "hey computer"
- `n` -> capture/upload one `not_wake` clip
- `h` -> show commands

Uploaded clips are saved to:

- `server/recordings/collected/<label>/...`
- `dataset/raw/authorized_user_wake/...` for authorized wake
- `dataset/raw/unknown_user_wake/...` for unknown-user wake
- `dataset/raw/unknown_speech/...` for `not_wake`
