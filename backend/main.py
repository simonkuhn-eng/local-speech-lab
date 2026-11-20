from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .asr_service import ASRService

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE_DIR / "model" / "parakeet-tdt-0.6b-v2.nemo"
DEFAULT_RECORDINGS_DIR = BASE_DIR / "data" / "recordings"
FRONTEND_DIR = BASE_DIR / "frontend"

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/vnd.wave",
}


class TranscriptionRequest(BaseModel):
    recording_id: str = Field(..., description="Identifier returned when the audio was uploaded")
    with_timestamps: bool = Field(True, description="Whether to include word-level timestamps")


class RecordingResponse(BaseModel):
    recording_id: str
    filename: str
    saved_at: datetime
    size_bytes: int


app = FastAPI(title="Local STT Service", version="0.1.0")
asr_service: Optional[ASRService] = None
recordings_dir: Path = DEFAULT_RECORDINGS_DIR


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # UI is served locally; loosened for convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    global asr_service, recordings_dir

    model_path = Path(os.getenv("ASR_MODEL_PATH", str(DEFAULT_MODEL))).expanduser()
    device = os.getenv("ASR_DEVICE", "cpu")
    batch_size = int(os.getenv("ASR_BATCH_SIZE", "1"))
    recordings_dir = Path(os.getenv("ASR_RECORDINGS_DIR", str(DEFAULT_RECORDINGS_DIR))).expanduser().resolve()
    recordings_dir.mkdir(parents=True, exist_ok=True)

    asr_service = ASRService(
        model_path=model_path,
        device=device,
        transcription_batch_size=batch_size,
    )

    # Warm up the model (ensures missing dependencies trigger during startup)
    await asr_service.ensure_loaded()

    if FRONTEND_DIR.exists():
        app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

    LOGGER.info("Service startup complete. Model: %s", model_path)


async def get_asr_service() -> ASRService:
    if asr_service is None:
        raise HTTPException(status_code=503, detail="ASR service not ready")
    return asr_service


@app.get("/api/health")
async def health() -> JSONResponse:
    if asr_service is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return JSONResponse(content={"status": "ok"})


@app.post("/api/recordings", response_model=RecordingResponse)
async def upload_recording(
    file: UploadFile = File(...),
) -> RecordingResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio type: {file.content_type}")

    recording_id = uuid4().hex
    filename = f"{recording_id}.wav"
    destination = recordings_dir / filename

    size = 0
    with destination.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            buffer.write(chunk)

    saved_at = datetime.utcnow()
    LOGGER.info("Stored audio %s (%s bytes) at %s", filename, size, destination)
    return RecordingResponse(recording_id=recording_id, filename=filename, saved_at=saved_at, size_bytes=size)


@app.post("/api/transcribe")
async def transcribe_audio(
    payload: TranscriptionRequest,
    service: ASRService = Depends(get_asr_service),
):
    audio_path = recordings_dir / f"{payload.recording_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    transcription = await service.transcribe(audio_path=audio_path, timestamps=payload.with_timestamps)
    return {
        "recording_id": payload.recording_id,
        "audio_path": audio_path.name,
        **transcription,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
