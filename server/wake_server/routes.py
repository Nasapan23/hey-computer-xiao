from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from .audio_io import safe_label, trim_pcm_to_int16, validate_pcm_headers, write_wav
from .config import DATASET_RAW_DIR, LABEL_TO_DATASET_DIR, RECORDINGS_DIR
from .state import state
from .transcription import load_transcription_sidecar, save_transcription_sidecar, whisper_transcriber

router = APIRouter()


class RetryTranscriptionRequest(BaseModel):
    path: str


def scan_recent_command_uploads(limit: int) -> list[dict]:
    commands_root = RECORDINGS_DIR / "commands"
    if not commands_root.exists():
        return []

    files = sorted(commands_root.rglob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    recent_entries: list[dict] = []
    for wav_path in files[:limit]:
        try:
            stat = wav_path.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            relative_path = wav_path.relative_to(RECORDINGS_DIR).as_posix()
            transcription = load_transcription_sidecar(wav_path) or {}
            recent_entries.append(
                {
                    "timestamp": timestamp,
                    "trigger_label": wav_path.parent.name,
                    "trigger_score": "",
                    "path": relative_path,
                    "sample_rate": 16000,
                    "bytes": stat.st_size,
                    "transcription_status": transcription.get("transcription_status", ""),
                    "transcript": transcription.get("transcript", ""),
                    "transcription_language": transcription.get("transcription_language", ""),
                    "transcription_error": transcription.get("transcription_error", ""),
                }
            )
        except OSError:
            continue
    return recent_entries


def transcribe_command_clip(wav_path: Path, relative_path: str) -> None:
    result = whisper_transcriber.transcribe_file(wav_path)
    try:
        save_transcription_sidecar(wav_path, result)
    except OSError as exc:
        existing_error = result.get("transcription_error", "").strip()
        sidecar_error = f"Failed to write transcription sidecar: {exc}"
        result["transcription_error"] = f"{existing_error}; {sidecar_error}".strip("; ").strip()

    state.set_command_transcription(
        relative_wav_path=relative_path,
        status=result.get("transcription_status", ""),
        transcript=result.get("transcript", ""),
        language=result.get("transcription_language", ""),
        error=result.get("transcription_error", ""),
    )


def resolve_command_wav_path(relative_path: str) -> tuple[Path, str]:
    normalized = relative_path.strip().lstrip("/\\")
    if not normalized:
        raise HTTPException(status_code=400, detail="Missing recording path")

    wav_path = (RECORDINGS_DIR / normalized).resolve()
    commands_root = (RECORDINGS_DIR / "commands").resolve()
    if wav_path.suffix.lower() != ".wav" or commands_root not in wav_path.parents:
        raise HTTPException(status_code=400, detail="Path must point to a command WAV recording")
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    return wav_path, wav_path.relative_to(RECORDINGS_DIR).as_posix()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui", status_code=307)


@router.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>XIAO Wake Server Monitor</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --card: #ffffff;
      --line: #d6deeb;
      --text: #0f172a;
      --muted: #475569;
      --accent: #0b6cff;
      --ok: #0f9d58;
      --warn: #d97706;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background: linear-gradient(160deg, #eef3ff 0%, #f9fbff 55%, #f4f7fb 100%);
    }
    .wrap {
      max-width: 1080px;
      margin: 28px auto 36px;
      padding: 0 14px;
    }
    .title {
      margin: 0 0 14px;
      font-size: 1.45rem;
      letter-spacing: 0.02em;
    }
    .meta {
      color: var(--muted);
      margin-bottom: 16px;
      font-size: 0.95rem;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: 0 2px 9px rgba(15, 23, 42, 0.04);
    }
    .card .k { color: var(--muted); font-size: 0.82rem; margin-bottom: 4px; }
    .card .v { font-size: 1.4rem; font-weight: 650; }
    .layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    @media (max-width: 940px) {
      .layout { grid-template-columns: 1fr; }
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 2px 9px rgba(15, 23, 42, 0.04);
    }
    .panel h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 1rem;
      border-bottom: 1px solid var(--line);
    }
    .list {
      max-height: 520px;
      overflow-y: auto;
    }
    .item {
      border-top: 1px solid var(--line);
      padding: 10px 14px;
      font-size: 0.92rem;
    }
    .item:first-child { border-top: 0; }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 6px;
    }
    .pill {
      border-radius: 999px;
      padding: 2px 9px;
      font-size: 0.78rem;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--muted);
    }
    .pill.ok {
      color: var(--ok);
      border-color: #b7e8cb;
      background: #ecf9f1;
    }
    .pill.warn {
      color: var(--warn);
      border-color: #f9ddb2;
      background: #fff7eb;
    }
    .path { color: var(--muted); font-family: ui-monospace, "Consolas", monospace; font-size: 0.8rem; word-break: break-all; }
    .transcript {
      margin-top: 8px;
      padding: 7px 9px;
      border-radius: 8px;
      border: 1px solid #d8e3f6;
      background: #f7faff;
      color: #1e293b;
      font-size: 0.86rem;
      line-height: 1.35;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .transcript.warn {
      border-color: #f4d7b5;
      background: #fff7ec;
      color: #8a5200;
    }
    .empty { padding: 16px 14px; color: var(--muted); font-size: 0.9rem; }
    audio { width: 100%; height: 32px; }
    button {
      border: 1px solid #bcd3ff;
      background: #f0f6ff;
      color: #164db3;
      border-radius: 8px;
      font-size: 0.85rem;
      padding: 6px 10px;
      cursor: pointer;
    }
    button:hover { background: #e7f0ff; }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.65;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">XIAO Wake Server Monitor</h1>
    <div class="meta">Shows wake signals and uploaded 5-second command clips. Auto-refresh every 2 seconds.</div>

    <div class="cards">
      <div class="card"><div class="k">Authorized Wake Count</div><div class="v" id="authorizedCount">0</div></div>
      <div class="card"><div class="k">Authorized Events</div><div class="v" id="authorizedEvents">0</div></div>
      <div class="card"><div class="k">Unknown User Events</div><div class="v" id="unknownEvents">0</div></div>
      <div class="card"><div class="k">Command Clips</div><div class="v" id="clipCount">0</div></div>
    </div>

    <div style="margin-bottom: 12px;">
      <button id="refreshBtn" type="button">Refresh Now</button>
    </div>

    <div class="layout">
      <section class="panel">
        <h2>Received Signals</h2>
        <div id="signalsList" class="list"></div>
      </section>
      <section class="panel">
        <h2>Recorded 5-Second Clips</h2>
        <div id="commandsList" class="list"></div>
      </section>
    </div>
  </div>

  <script>
    const signalsList = document.getElementById("signalsList");
    const commandsList = document.getElementById("commandsList");
    const authorizedCount = document.getElementById("authorizedCount");
    const authorizedEvents = document.getElementById("authorizedEvents");
    const unknownEvents = document.getElementById("unknownEvents");
    const clipCount = document.getElementById("clipCount");
    const refreshBtn = document.getElementById("refreshBtn");
    const retryableTranscriptStatuses = new Set(["no_speech", "error", "unavailable"]);
    let lastSignalsSignature = null;
    let lastCommandsSignature = null;
    let deferredCommandsRender = false;

    function safeText(value) {
      if (value === undefined || value === null || value === "") return "n/a";
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function safeMaybeText(value) {
      if (value === undefined || value === null) return "";
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatTimestamp(value) {
      if (!value) return "n/a";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString();
    }

    function buildSignalsSignature(items) {
      return (items || []).map((item) => [
        safeText(item.timestamp),
        safeText(item.event),
        safeText(item.score),
        item.authorized ? "1" : "0",
      ].join("|")).join("||");
    }

    function buildCommandsSignature(items) {
      return (items || []).map((item) => [
        safeText(item.path),
        safeText(item.timestamp),
        safeText(item.bytes),
        safeMaybeText(item.transcription_status),
        safeMaybeText(item.transcript),
        safeMaybeText(item.transcription_error),
      ].join("|")).join("||");
    }

    function isAnyCommandClipPlaying() {
      const clips = Array.from(commandsList.querySelectorAll("audio"));
      return clips.some((clip) => !clip.paused && !clip.ended);
    }

    function renderSignals(items) {
      if (!items || items.length === 0) {
        signalsList.innerHTML = '<div class="empty">No signals received yet.</div>';
        return;
      }

      signalsList.innerHTML = items.map((item) => {
        const isAuthorized = !!item.authorized;
        const pillClass = isAuthorized ? "pill ok" : "pill warn";
        const label = isAuthorized ? "authorized" : "rejected";
        return `
          <article class="item">
            <div class="row">
              <span class="${pillClass}">${label}</span>
              <span class="pill">${safeText(item.event)}</span>
              <span class="pill">score: ${safeText(item.score)}</span>
            </div>
            <div>${formatTimestamp(item.timestamp)}</div>
          </article>
        `;
      }).join("");
    }

    function renderCommands(items) {
      if (!items || items.length === 0) {
        commandsList.innerHTML = '<div class="empty">No command clips uploaded yet.</div>';
        return;
      }

      commandsList.innerHTML = items.map((item) => {
        const statusRaw = String(item.transcription_status || "").trim() || "pending";
        const status = safeMaybeText(statusRaw);
        const transcript = safeMaybeText(item.transcript);
        const transcriptionError = safeMaybeText(item.transcription_error);
        const language = safeMaybeText(item.transcription_language);
        const transcriptPillClass = status === "ready" ? "pill ok" : (status === "error" || status === "unavailable" ? "pill warn" : "pill");
        const transcriptTextClass = status === "error" || status === "unavailable" ? "transcript warn" : "transcript";
        const transcriptBody = transcript || transcriptionError || (status === "pending" ? "Transcription in progress..." : "No speech detected.");
        const languageLabel = language ? `, ${language}` : "";
        const canRetry = retryableTranscriptStatuses.has(statusRaw);
        const retryButton = canRetry ? `<button type="button" class="retry-btn" data-path="${safeText(item.path)}">Retry transcription</button>` : "";
        return `
        <article class="item">
          <div class="row">
            <span class="pill ok">${safeText(item.trigger_label)}</span>
            <span class="pill">score: ${safeText(item.trigger_score)}</span>
            <span class="pill">${safeText(item.sample_rate)} Hz</span>
            <span class="${transcriptPillClass}">transcript: ${safeText(status)}${safeMaybeText(languageLabel)}</span>
          </div>
          <div style="margin-bottom: 6px;">${formatTimestamp(item.timestamp)}</div>
          <audio controls preload="none" src="${safeText(item.url)}"></audio>
          <div class="${transcriptTextClass}">${safeText(transcriptBody)}</div>
          <div style="margin-top: 8px;">${retryButton}</div>
          <div class="path">${safeText(item.path)}</div>
        </article>
      `;
      }).join("");
    }

    async function retryTranscription(path, button) {
      const response = await fetch("/api/transcription/retry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || ("Request failed: " + response.status));
      }
      await refresh();
      if (button instanceof HTMLButtonElement) {
        button.textContent = "Retry queued";
      }
    }

    async function refresh() {
      try {
        const response = await fetch("/api/activity?limit=120", { cache: "no-store" });
        if (!response.ok) throw new Error("Request failed: " + response.status);
        const data = await response.json();

        authorizedCount.textContent = safeText(data.counters?.authorized_wake_count ?? 0);
        authorizedEvents.textContent = safeText(data.counters?.events?.authorized_user_wake ?? 0);
        unknownEvents.textContent = safeText(data.counters?.events?.unknown_user_wake ?? 0);
        const signals = data.signals || [];
        const commands = data.commands || [];
        clipCount.textContent = safeText(commands.length);

        const signalsSignature = buildSignalsSignature(signals);
        if (signalsSignature !== lastSignalsSignature) {
          renderSignals(signals);
          lastSignalsSignature = signalsSignature;
        }

        const commandsSignature = buildCommandsSignature(commands);
        if (isAnyCommandClipPlaying()) {
          if (commandsSignature !== lastCommandsSignature) {
            deferredCommandsRender = true;
          }
        } else if (commandsSignature !== lastCommandsSignature || deferredCommandsRender) {
          renderCommands(commands);
          lastCommandsSignature = commandsSignature;
          deferredCommandsRender = false;
        }
      } catch (error) {
        signalsList.innerHTML = `<div class="empty">Failed to fetch activity: ${safeText(error.message)}</div>`;
      }
    }

    refreshBtn.addEventListener("click", refresh);
    commandsList.addEventListener("click", async (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const button = target.closest("button.retry-btn");
      if (!(button instanceof HTMLButtonElement)) return;
      const path = button.dataset.path || "";
      if (!path) return;
      const originalLabel = button.textContent || "Retry transcription";
      button.disabled = true;
      button.textContent = "Retrying...";
      try {
        await retryTranscription(path, button);
      } catch (error) {
        button.textContent = originalLabel;
        alert("Retry failed: " + (error && error.message ? error.message : String(error)));
      } finally {
        button.disabled = false;
      }
    });
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


@router.post("/upload")
async def upload_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    x_audio_sample_rate: int = Header(default=16000),
    x_audio_bits_per_sample: int = Header(default=16),
    x_audio_channels: int = Header(default=1),
    x_trigger_label: str = Header(default="authorized_user_wake"),
    x_trigger_score: str = Header(default=""),
) -> dict[str, str | int]:
    pcm = await request.body()
    if not pcm:
        raise HTTPException(status_code=400, detail="Empty audio body")

    validate_pcm_headers(x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)
    pcm = trim_pcm_to_int16(pcm)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    trigger_label = safe_label(x_trigger_label)
    wav_path = RECORDINGS_DIR / "commands" / trigger_label / f"command_{timestamp}.wav"
    write_wav(wav_path, pcm, x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)
    relative_path = wav_path.relative_to(RECORDINGS_DIR).as_posix()
    state.record_command_upload(
        timestamp=timestamp_iso,
        trigger_label=trigger_label,
        trigger_score=x_trigger_score,
        relative_wav_path=relative_path,
        sample_rate=x_audio_sample_rate,
        byte_count=len(pcm),
    )
    background_tasks.add_task(transcribe_command_clip, wav_path, relative_path)

    return {
        "status": "ok",
        "saved_as": str(wav_path),
        "recording_url": f"/recordings/{relative_path}",
        "trigger_label": trigger_label,
        "trigger_score": x_trigger_score,
        "bytes": len(pcm),
        "sample_rate": x_audio_sample_rate,
        "bits_per_sample": x_audio_bits_per_sample,
        "channels": x_audio_channels,
        "transcription_status": "pending",
    }


@router.post("/api/transcription/retry")
def retry_transcription(payload: RetryTranscriptionRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    wav_path, relative_path = resolve_command_wav_path(payload.path)
    state.set_command_transcription(
        relative_wav_path=relative_path,
        status="pending",
        transcript="",
        language="",
        error="",
    )
    background_tasks.add_task(transcribe_command_clip, wav_path, relative_path)
    return {
        "status": "queued",
        "path": relative_path,
    }


@router.post("/collect")
async def collect_labeled_audio(
    request: Request,
    x_clip_label: str = Header(default="not_wake"),
    x_audio_sample_rate: int = Header(default=16000),
    x_audio_bits_per_sample: int = Header(default=16),
    x_audio_channels: int = Header(default=1),
) -> dict[str, str | int]:
    pcm = await request.body()
    if not pcm:
        raise HTTPException(status_code=400, detail="Empty audio body")

    label = x_clip_label.strip().lower()
    if label not in LABEL_TO_DATASET_DIR:
        labels = ", ".join(sorted(LABEL_TO_DATASET_DIR))
        raise HTTPException(status_code=400, detail=f"Label must be one of: {labels}")

    validate_pcm_headers(x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)
    pcm = trim_pcm_to_int16(pcm)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    clip_number = state.next_collect_number()

    recordings_path = RECORDINGS_DIR / "collected" / label / f"{label}_{timestamp}_{clip_number:05d}.wav"
    dataset_subdir = LABEL_TO_DATASET_DIR[label]
    dataset_path = DATASET_RAW_DIR / dataset_subdir / f"xiao_{label}_{timestamp}_{clip_number:05d}.wav"

    write_wav(recordings_path, pcm, x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)
    write_wav(dataset_path, pcm, x_audio_sample_rate, x_audio_bits_per_sample, x_audio_channels)

    return {
        "status": "ok",
        "label": label,
        "clip_number": clip_number,
        "saved_recording": str(recordings_path),
        "saved_dataset": str(dataset_path),
        "bytes": len(pcm),
        "sample_rate": x_audio_sample_rate,
    }


@router.post("/ping")
async def wake_ping(request: Request) -> dict[str, str | int]:
    body = await request.body()
    event = "authorized_user_wake"
    score = ""
    authorized = False
    if body:
        try:
            payload = await request.json()
            event = str(payload.get("event", event))
            score = str(payload.get("score", ""))
            authorized = bool(payload.get("authorized", event == "authorized_user_wake"))
        except Exception:
            pass

    event = safe_label(event)
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    count, event_count = state.record_detection(event, authorized, score, timestamp_iso)
    return {
        "status": "ok",
        "event": event,
        "count": count,
        "event_count": event_count,
        "score": score,
        "timestamp": timestamp_iso,
    }


@router.get("/counter")
def read_counter() -> dict[str, int | dict[str, int]]:
    return state.read_counters()


@router.get("/api/activity")
def read_activity(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    signals = state.read_recent_signals(limit=limit)
    commands_memory = state.read_recent_command_uploads(limit=limit)
    commands_disk = scan_recent_command_uploads(limit=limit)

    merged_by_path: dict[str, dict] = {}
    for entry in commands_disk + commands_memory:
        existing = merged_by_path.get(entry["path"])
        if existing is None or entry["timestamp"] > existing["timestamp"]:
            merged_by_path[entry["path"]] = entry

    commands = sorted(merged_by_path.values(), key=lambda entry: entry["timestamp"], reverse=True)[:limit]
    commands_with_urls = [
        {
            **entry,
            "transcription_status": entry.get("transcription_status", ""),
            "transcript": entry.get("transcript", ""),
            "transcription_language": entry.get("transcription_language", ""),
            "transcription_error": entry.get("transcription_error", ""),
            "url": f"/recordings/{entry['path']}",
        }
        for entry in commands
    ]
    return {
        "counters": state.read_counters(),
        "signals": signals,
        "commands": commands_with_urls,
    }
