from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
import soundfile as sf

from .asr_service import ASRService

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE_DIR / "model" / "parakeet-tdt-0.6b-v2.nemo"
DEFAULT_RECORDINGS_DIR = BASE_DIR / "data" / "recordings"

SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("ASR_CHANNELS", "1"))
POLL_INTERVAL = float(os.getenv("ASR_HOTKEY_POLL_SECONDS", "0.02"))
MIN_CLIP_SECONDS = float(os.getenv("ASR_MIN_CLIP_SECONDS", "0.35"))

HOTKEY_DESCRIPTION = "AltGr+P"
TRIGGER_KEY_NAMES: Sequence[str] = ("p", "key p")
ALTGR_KEY_NAMES: Sequence[str] = (
    "alt gr",
    "altgr",
    "altgraph",
    "alt_graph",
    "right alt",
    "alt_r",
)


class AudioRecorder:
    """Capture a mono audio clip while the hotkey is pressed."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream: sd.InputStream | None = None
        self._frames: List[np.ndarray] = []

    def start(self) -> bool:
        if self._stream is not None:
            return False

        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        LOGGER.debug("Recorder stream started")
        return True

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.empty((0, self.channels), dtype=np.float32)

        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None

        if not self._frames:
            return np.empty((0, self.channels), dtype=np.float32)

        data = np.concatenate(self._frames, axis=0)
        self._frames = []
        LOGGER.debug("Recorder captured %s samples", data.shape[0])
        return data

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None
        self._frames = []

    def _callback(self, indata: np.ndarray, frames: int, _time: float, status: sd.CallbackFlags) -> None:
        if status:
            LOGGER.warning("Recorder stream status: %s", status)
        self._frames.append(indata.copy())


@dataclass
class DictationState:
    recording: bool = False
    transcribing: bool = False
    last_started: float = 0.0


def _key_to_scan_codes(name: str) -> Set[int]:
    try:
        codes = keyboard.key_to_scan_codes(name, error_if_missing=False)  # type: ignore[arg-type]
    except TypeError:
        # Older keyboard versions use positional-only signature.
        codes = keyboard.key_to_scan_codes(name)
    if not codes:
        return set()
    if isinstance(codes, int):
        return {codes}
    return set(codes)


def gather_scancodes(
    names: Iterable[str],
    *,
    fallback: str | None = None,
    label: str = "key",
) -> Set[int]:
    scancodes: Set[int] = set()
    for name in names:
        try:
            scancodes.update(_key_to_scan_codes(name))
        except Exception:  # pragma: no cover - defensive fallback
            LOGGER.debug("Unable to resolve scan code for key '%s'", name)
    if not scancodes and fallback:
        LOGGER.warning("Fell back to default scan code for %s; detection may be unreliable", label)
        try:
            scancodes.update(_key_to_scan_codes(fallback))
        except Exception:  # pragma: no cover - defensive fallback
            LOGGER.debug("Fallback key '%s' also failed", fallback)
    return scancodes


TRIGGER_SCANCODES = gather_scancodes(TRIGGER_KEY_NAMES, fallback="p", label="trigger key")
ALTGR_SCANCODES = gather_scancodes(ALTGR_KEY_NAMES, fallback="alt gr", label="AltGr modifier")


def is_trigger_pressed() -> bool:
    for code in TRIGGER_SCANCODES:
        try:
            if keyboard.is_pressed(code):
                return True
        except ValueError:
            continue
    return False


def is_altgr_pressed() -> bool:
    for code in ALTGR_SCANCODES:
        try:
            if keyboard.is_pressed(code):
                return True
        except ValueError:
            continue
    return False


def is_combo_active() -> bool:
    altgr_active = is_altgr_pressed()
    ctrl_alt_active = keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt")
    return (altgr_active or ctrl_alt_active) and is_trigger_pressed()


async def transcription_loop(
    service: ASRService,
    recorder: AudioRecorder,
    recordings_dir: Path,
) -> None:
    state = DictationState()
    LOGGER.info("Listening for %s — hold to dictate, release to transcribe", HOTKEY_DESCRIPTION)

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            combo_active = is_combo_active()

            if combo_active and not state.recording:
                if state.transcribing:
                    LOGGER.info("Hotkey pressed while transcription in progress; waiting to finish")
                    continue
                if recorder.start():
                    state.recording = True
                    state.last_started = time.time()
                    LOGGER.info("Dictation started")
                continue

            if not combo_active and state.recording:
                state.recording = False
                await handle_clip(service, recorder, recordings_dir, state)
    finally:
        recorder.close()


async def handle_clip(
    service: ASRService,
    recorder: AudioRecorder,
    recordings_dir: Path,
    state: DictationState,
) -> None:
    audio = recorder.stop()
    if audio.size == 0:
        LOGGER.info("Clip discarded: no audio captured")
        return

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    duration = audio.shape[0] / recorder.sample_rate
    if duration < MIN_CLIP_SECONDS:
        LOGGER.info("Clip discarded: %.2fs shorter than minimum %.2fs", duration, MIN_CLIP_SECONDS)
        return

    state.transcribing = True
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    recordings_dir.mkdir(parents=True, exist_ok=True)
    audio_path = recordings_dir / f"dictation-{timestamp}.wav"

    try:
        sf.write(str(audio_path), audio, recorder.sample_rate)
        LOGGER.info("Saved temp clip %s (%.2fs)", audio_path.name, duration)

        result = await service.transcribe(audio_path=audio_path, timestamps=False)
        text = (result.get("text") or "").strip()
        if text:
            try:
                pyperclip.copy(text)
                LOGGER.info("Transcript copied to clipboard (%d chars)", len(text))
            except pyperclip.PyperclipException as exc:  # pragma: no cover - clipboard failures depend on host
                LOGGER.warning("Clipboard copy failed: %s", exc)
        else:
            LOGGER.info("Transcription complete, but text was empty")
    except Exception as exc:  # pragma: no cover - runtime safety
        LOGGER.exception("Failed to transcribe clip: %s", exc)
    finally:
        try:
            audio_path.unlink(missing_ok=True)
            LOGGER.info("Deleted temp audio %s", audio_path.name)
        except OSError:
            LOGGER.warning("Unable to delete temp audio %s", audio_path)
        state.transcribing = False


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("ASR_LOG_LEVEL", "INFO").upper(),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    model_path = Path(os.getenv("ASR_MODEL_PATH", str(DEFAULT_MODEL))).expanduser()
    device = os.getenv("ASR_DEVICE", "cpu")
    batch_size = int(os.getenv("ASR_BATCH_SIZE", "1"))
    recordings_dir = Path(os.getenv("ASR_RECORDINGS_DIR", str(DEFAULT_RECORDINGS_DIR))).expanduser()

    LOGGER.info("Loading ASR model from %s on device %s", model_path, device)
    service = ASRService(model_path=model_path, device=device, transcription_batch_size=batch_size)
    await service.ensure_loaded()
    LOGGER.info("Model ready. Background dictation service running.")

    recorder = AudioRecorder(sample_rate=SAMPLE_RATE, channels=CHANNELS)
    await transcription_loop(service, recorder, recordings_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Dictation service interrupted by user")