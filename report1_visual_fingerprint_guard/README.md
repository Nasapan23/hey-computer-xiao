# Report 1 Starter: Visual Fingerprint Guard (Camera-Only)

This is a simple and slightly unique Report 1 idea:

- Train a model on the **normal visual fingerprint** of one fixed scene (for example, your desk or room corner).
- Use the XIAO ESP32S3 Sense camera (MicroPython) to stream snapshots.
- Run ML on the PC to detect when the scene looks unusual (object added, person enters, camera moved, sudden lighting change).

It is camera-only, easy to implement, and different from the lab support examples.

## Why This Fits Report 1

1. Problem definition + dataset: normal scene snapshots collected from XIAO camera.
2. ML model used: PCA reconstruction error anomaly detector.
3. Training results: threshold and training error distribution.
4. Hardware used: XIAO ESP32S3 Sense only.
5. MicroPython explanations: camera HTTP server + status endpoint.
6. Results on real data: live NORMAL/ABNORMAL decisions.
7. Conclusion: robustness and limits.

## Folder Layout

```text
report1_visual_fingerprint_guard/
|-- micropython/
|   |-- main.py
|   `-- secrets.example.py
|-- pc/
|   |-- requirements.txt
|   |-- common.py
|   |-- collect_normal_frames.py
|   |-- train_fingerprint_model.py
|   `-- live_guard.py
|-- data/
|   |-- normal_frames/
|   `-- models/
`-- report/
    |-- report_template.md
    `-- live_results.csv
```

## Step 1 - Flash + MicroPython

Use your normal XIAO ESP32S3 Sense MicroPython setup flow (BootLoader + UF2 + Thonny).

Copy `micropython/main.py` to the board as `/main.py`.

Create `micropython/secrets.py` on board from `secrets.example.py` and set Wi-Fi credentials.

### Quick Flash (already prepared in this repo)

Firmware bundle path:

```text
report1_visual_fingerprint_guard/firmware_bundle/XIAO ESP32S3 Micropython/firmware.bin
```

One-command flashing script:

```powershell
powershell -ExecutionPolicy Bypass -File report1_visual_fingerprint_guard\tools\flash_micropython.ps1 -Port COM11
```

Replace `COM11` with your board port.

### If Camera Init Fails With "Detected camera not supported"

Some newer XIAO ESP32S3 Sense boards use OV3660 sensor (instead of OV2640).  
If camera is detected on SCCB (`0x3c`) but init still fails, flash the OV3660-capable camera firmware:

```powershell
powershell -ExecutionPolicy Bypass -File report1_visual_fingerprint_guard\tools\flash_micropython.ps1 -Port COM11 -FirmwarePath report1_visual_fingerprint_guard\firmware_bundle\mpy_cam_xiao\firmware.bin
```

Then re-upload `micropython/main.py` and `micropython/secrets.py`.

## Step 2 - Start Camera Server on Board

After reset, Serial should print the board IP. Endpoints:

- `GET /capture` -> JPEG snapshot
- `GET /health` -> current board status
- `POST /status` -> receives NORMAL/ABNORMAL from PC
- `GET /` -> simple status page

## Step 3 - Set Up PC Environment

From repo root:

```powershell
python -m venv report1_visual_fingerprint_guard\pc\.venv
report1_visual_fingerprint_guard\pc\.venv\Scripts\activate
pip install -r report1_visual_fingerprint_guard\pc\requirements.txt
```

## Step 4 - Collect Normal Frames

Use a stable scene that represents NORMAL.

```powershell
python report1_visual_fingerprint_guard\pc\collect_normal_frames.py --board-ip 192.168.1.50 --samples 150
```

## Step 5 - Train Model

```powershell
python report1_visual_fingerprint_guard\pc\train_fingerprint_model.py --frames-dir report1_visual_fingerprint_guard\data\normal_frames
```

This writes:

- `data/models/fingerprint_guard.joblib`
- `data/models/training_summary.json`

## Step 6 - Run Live Guard

```powershell
python report1_visual_fingerprint_guard\pc\live_guard.py --board-ip 192.168.1.50 --show
```

Press `q` to stop live mode.

## Suggested Demo

1. Keep scene as trained baseline -> mostly NORMAL.
2. Enter scene or place an object suddenly -> ABNORMAL.
3. Remove object / restore scene -> returns to NORMAL.
