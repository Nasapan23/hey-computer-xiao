# PC Wake Counter

This is the PC side of the experiment.

The ESP32 sends a wake-word ping to:

```text
POST /ping
```

The server increments an in-memory counter every time it receives a ping.

Read counter value here:

```text
GET /counter
```

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

## Test Without ESP32

With the server running, open another terminal:

```powershell
curl -X POST http://127.0.0.1:8000/ping
curl http://127.0.0.1:8000/counter
```

## Labeled Data Collection for Training

The server also supports labeled clip uploads from XIAO:

```text
POST /collect
```

Required header:

- `X-Clip-Label: wake` or `X-Clip-Label: not_wake`

Audio format must be:

- raw PCM body
- 16-bit
- mono
- sample rate in `X-Audio-Sample-Rate` header

When accepted, the server saves each clip to:

- `server/recordings/collected/<label>/...`
- `dataset/raw/wake_word/...` for `wake`
- `dataset/raw/unknown_speech/...` for `not_wake`
