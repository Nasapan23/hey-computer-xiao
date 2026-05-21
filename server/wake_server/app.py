from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import RECORDINGS_DIR
from .routes import router

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="XIAO ESP32S3 Wake-Word Audio Receiver")
app.mount("/recordings", StaticFiles(directory=str(RECORDINGS_DIR)), name="recordings")
app.include_router(router)
