# PC Wake Counter

This is the PC side of the experiment.

The ESP32 sends detection events to:

```text
POST /ping
```

The server increments in-memory counters for authorized and unknown-user wake events.

Read counter value here:

```text
GET /counter
```

Open the live monitor UI here:

```text
GET /ui
```

The root path `/` redirects to `/ui`.

## Run on Windows

Double-click:

```text
run_server.bat
```

The server listens on:

```text
http://0.0.0.0:8000
```

Use your PC's real LAN IP in the ESP32 sketch, for example:

```cpp
const char *PC_SERVER_PING_URL = "http://192.168.1.50:8000/ping";
```

## Code Layout

```text
server/
|-- main.py             FastAPI entry point used by uvicorn
|-- run_server.bat
|-- test_upload.py
`-- wake_server/
    |-- app.py          creates the FastAPI app
    |-- routes.py       /health, /ping, /upload, /collect, /counter, /ui, /api/activity
    |-- audio_io.py     PCM validation and WAV writing
    |-- transcription.py optional local Whisper transcription for uploaded command clips
    |-- config.py       paths and label mapping
    `-- state.py        in-memory counters
```

## Test Without ESP32

With the server running, open another terminal:

```powershell
curl -X POST http://127.0.0.1:8000/ping
curl http://127.0.0.1:8000/counter
```

## Command Audio Uploads

After an authorized wake, the ESP32 records the next 5 seconds and uploads raw PCM audio to:

```text
POST /upload
```

The server saves command clips under:

```text
server/recordings/commands/<trigger_label>/
```

Those files are also exposed for browser playback under:

```text
/recordings/<relative_path_inside_recordings>
```

The server also starts a background transcription task for each uploaded command clip.
Transcripts are shown in `/ui` and exposed in `/api/activity`.

Transcription sidecar files are saved next to each command WAV:

```text
server/recordings/commands/<trigger_label>/command_<timestamp>.transcript.json
```

Environment variables for transcription (optional):

- `WAKE_WHISPER_MODEL` (default: `tiny.en`)
- `WAKE_WHISPER_DEVICE` (default: `cpu`)
- `WAKE_WHISPER_COMPUTE_TYPE` (default: `int8`)
- `WAKE_WHISPER_LANGUAGE` (default: `en`)

## Labeled Data Collection for Training

The server also supports labeled clip uploads from XIAO:

```text
POST /collect
```

Required header:

- `X-Clip-Label: authorized_user_wake`
- `X-Clip-Label: unknown_user_wake`
- `X-Clip-Label: not_wake`

Audio format must be:

- raw PCM body
- 16-bit
- mono
- sample rate in `X-Audio-Sample-Rate` header

When accepted, the server saves each clip to:

- `server/recordings/collected/<label>/...`
- `dataset/raw/authorized_user_wake/...` for authorized wake
- `dataset/raw/unknown_user_wake/...` for unknown-user wake
- `dataset/raw/unknown_speech/...` for `not_wake`
