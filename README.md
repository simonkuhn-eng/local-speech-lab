# Local Speech Lab

Offline speech-to-text service powered by FastAPI and NVIDIA NeMo (`parakeet-tdt-0.6b-v2`), with a browser UI and a push-to-talk Windows hotkey daemon.

Audio recordings and transcription stay completely on the local machine.

## Features

- FastAPI backend (`backend/main.py`) with WAV upload and NeMo ASR transcription endpoints, including optional word-level timestamps.
- Static browser frontend (`frontend/`) for microphone recording, playback, and transcript copy.
- Optional Windows push-to-talk daemon (`backend/hotkey_daemon.py`): hold `AltGr+P` while speaking to transcribe and copy text to the clipboard.
- Configurable device (`ASR_DEVICE=cpu` or `cuda`), model path (`ASR_MODEL_PATH`), and batch size (`ASR_BATCH_SIZE`).

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Place a local NeMo `.nemo` checkpoint at `model/parakeet-tdt-0.6b-v2.nemo` (or set `ASR_MODEL_PATH` to your checkpoint path). Model weights are not included in this repository.

## Running

Start the FastAPI server and UI (`http://localhost:8000/ui/`):

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Or launch both the API server and push-to-talk daemon on Windows via PowerShell:

```powershell
.\scripts\start-local-stt.ps1 -LaunchBrowser
```
