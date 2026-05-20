from datetime import datetime
from pathlib import Path
from threading import Lock
import wave

from fastapi import FastAPI, Header, HTTPException, Request


RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_RAW_DIR = Path(__file__).resolve().parents[1] / "dataset" / "raw"
LABEL_TO_DATASET_DIR = {
    "wake": "wake_word",
    "not_wake": "unknown_speech",
}

app = FastAPI(title="XIAO ESP32S3 Wake-Word Audio Receiver")
counter_lock = Lock()
wake_word_counter = 0
collect_counter_lock = Lock()
collect_counter = 0


def _validate_pcm_headers(sample_rate: int, bits_per_sample: int, channels: int) -> None:
    if bits_per_sample != 16:
        raise HTTPException(status_code=400, detail="Only 16-bit PCM is supported")
    if channels != 1:
        raise HTTPException(status_code=400, detail="Only mono PCM is supported")
    if sample_rate <= 0:
        raise HTTPException(status_code=400, detail="Invalid sample rate")


def _trim_pcm_to_int16(pcm: bytes) -> bytes:
    if len(pcm) % 2 != 0:
        return pcm[:-1]
    return pcm


def _write_wav(path: Path, pcm: bytes, sample_rate: int, bits_per_sample: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_audio(
    request: Request,
    x_audio_sample_rate: int = Header(default=16000),
    x_audio_bits_per_sample: int = Header(default=16),
    x_audio_channels: int = Header(default=1),
) -> dict[str, str | int]:
    pcm = await request.body()

    if not pcm:
        raise HTTPException(status_code=400, detail="Empty audio body")

    _validate_pcm_headers(x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)
    pcm = _trim_pcm_to_int16(pcm)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_path = RECORDINGS_DIR / f"wake_{timestamp}.wav"
    _write_wav(wav_path, pcm, x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)

    return {
        "saved_as": str(wav_path),
        "bytes": len(pcm),
        "sample_rate": x_audio_sample_rate,
        "bits_per_sample": x_audio_bits_per_sample,
        "channels": x_audio_channels,
    }


@app.post("/collect")
async def collect_labeled_audio(
    request: Request,
    x_clip_label: str = Header(default="not_wake"),
    x_audio_sample_rate: int = Header(default=16000),
    x_audio_bits_per_sample: int = Header(default=16),
    x_audio_channels: int = Header(default=1),
) -> dict[str, str | int]:
    global collect_counter

    pcm = await request.body()
    if not pcm:
        raise HTTPException(status_code=400, detail="Empty audio body")

    label = x_clip_label.strip().lower()
    if label not in LABEL_TO_DATASET_DIR:
        raise HTTPException(status_code=400, detail="Label must be one of: wake, not_wake")

    _validate_pcm_headers(x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)
    pcm = _trim_pcm_to_int16(pcm)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with collect_counter_lock:
        collect_counter += 1
        clip_number = collect_counter

    recordings_path = RECORDINGS_DIR / "collected" / label / f"{label}_{timestamp}_{clip_number:05d}.wav"
    dataset_subdir = LABEL_TO_DATASET_DIR[label]
    dataset_path = DATASET_RAW_DIR / dataset_subdir / f"xiao_{label}_{timestamp}_{clip_number:05d}.wav"

    _write_wav(recordings_path, pcm, x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)
    _write_wav(dataset_path, pcm, x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)

    return {
        "status": "ok",
        "label": label,
        "clip_number": clip_number,
        "saved_recording": str(recordings_path),
        "saved_dataset": str(dataset_path),
        "bytes": len(pcm),
        "sample_rate": x_audio_sample_rate,
    }


@app.post("/ping")
async def wake_ping(request: Request) -> dict[str, str | int]:
    global wake_word_counter

    body = await request.body()
    event = "wake_word"
    score = ""
    if body:
        try:
            payload = await request.json()
            event = str(payload.get("event", event))
            score = str(payload.get("score", ""))
        except Exception:
            pass

    with counter_lock:
        wake_word_counter += 1
        count = wake_word_counter

    return {
        "status": "ok",
        "event": event,
        "count": count,
        "score": score,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/counter")
def read_counter() -> dict[str, int]:
    with counter_lock:
        count = wake_word_counter
    return {"wake_word_count": count}
