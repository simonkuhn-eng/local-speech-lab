from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    import nemo.collections.asr as nemo_asr
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "nemo_toolkit is required for ASRService. Install it with `pip install nemo_toolkit[asr]`."
    ) from exc

LOGGER = logging.getLogger(__name__)


class ASRService:
    """Thin wrapper around a NeMo ASR model checkpoint."""

    def __init__(
        self,
        model_path: Path | str,
        device: str = "cpu",
        transcription_batch_size: int = 1,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ASR model checkpoint not found at {self.model_path}")

        self.device = torch.device(device)
        self.batch_size = transcription_batch_size
        self._model: Optional[nemo_asr.models.ASRModel] = None
        self._model_lock = asyncio.Lock()

    async def ensure_loaded(self) -> nemo_asr.models.ASRModel:
        """Lazily restore the NeMo checkpoint on first use."""
        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is not None:  # another waiter finished loading
                return self._model

            LOGGER.info("Loading NeMo ASR model from %s (device: %s)", self.model_path, self.device)
            with torch.inference_mode():
                model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=str(self.model_path),
                    map_location=str(self.device),
                )
                model.eval()
                model = model.to(self.device)

            self._model = model
            LOGGER.info("NeMo ASR model loaded successfully")
            return model

    async def transcribe(
        self,
        audio_path: Path | str,
        timestamps: bool = True,
    ) -> Dict[str, Any]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = await self.ensure_loaded()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._run_transcription(model=model, audio_path=audio_path, timestamps=timestamps),
        )
        return result

    def _run_transcription(
        self,
        model: nemo_asr.models.ASRModel,
        audio_path: Path,
        timestamps: bool,
    ) -> Dict[str, Any]:
        transcribe_kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "return_hypotheses": True,
        }

        LOGGER.debug("Running ASR on %s", audio_path)
        try:
            hypotheses = model.transcribe([str(audio_path)], **transcribe_kwargs)
        except TypeError as exc:  # pragma: no cover - compatibility fallback
            LOGGER.warning("Retrying transcription with legacy signature due to: %s", exc)
            legacy_kwargs = {
                "paths2audio_files": [str(audio_path)],
                "batch_size": self.batch_size,
                "return_hypotheses": True,
            }
            hypotheses = model.transcribe(**legacy_kwargs)
        if not hypotheses:
            return {"text": "", "words": [], "segments": []}

        hypothesis = hypotheses[0]
        payload = self._hypothesis_to_payload(hypothesis)
        payload.setdefault("text", "")
        if timestamps and not payload.get("words"):
            LOGGER.warning("Word-level timestamps not provided by current NeMo build; returning text only")
        return payload

    def _hypothesis_to_payload(self, hypothesis: Any) -> Dict[str, Any]:
        if hasattr(hypothesis, "to_dict"):
            data = hypothesis.to_dict()  # type: ignore[assignment]
        elif isinstance(hypothesis, dict):
            data = dict(hypothesis)
        else:
            text = getattr(hypothesis, "text", str(hypothesis))
            data = {"text": text}
            words = getattr(hypothesis, "words", None)
            segments = getattr(hypothesis, "segments", None)
            if words is not None:
                data["words"] = words
            if segments is not None:
                data["segments"] = segments

        normalized = {
            "text": data.get("text", ""),
            "words": self._normalize_word_timestamps(data),
            "segments": self._normalize_segment_timestamps(data),
        }
        return normalized

    def _normalize_word_timestamps(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_words = (
            data.get("words")
            or data.get("word_timestamps")
            or data.get("word")
            or data.get("words_timestamps")
        )
        if raw_words is None:
            return []

        words: List[Dict[str, Any]] = []
        for item in raw_words:
            if hasattr(item, "to_dict"):
                word_dict = item.to_dict()  # type: ignore[assignment]
            elif isinstance(item, dict):
                word_dict = dict(item)
            else:
                word_dict = {
                    "word": getattr(item, "word", getattr(item, "text", str(item))),
                    "start_time": getattr(item, "start_offset", getattr(item, "start_time", None)),
                    "end_time": getattr(item, "end_offset", getattr(item, "end_time", None)),
                    "confidence": getattr(item, "confidence", None),
                }

            word_entry = {
                "word": word_dict.get("word") or word_dict.get("text"),
                "start": self._extract_timestamp(word_dict, "start"),
                "end": self._extract_timestamp(word_dict, "end"),
            }
            confidence = word_dict.get("confidence")
            if confidence is not None:
                word_entry["confidence"] = confidence
            words.append(word_entry)
        return words

    def _normalize_segment_timestamps(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_segments = data.get("segments") or data.get("segment_timestamps") or data.get("segment")
        if raw_segments is None:
            return []

        segments: List[Dict[str, Any]] = []
        for item in raw_segments:
            if hasattr(item, "to_dict"):
                seg_dict = item.to_dict()  # type: ignore[assignment]
            elif isinstance(item, dict):
                seg_dict = dict(item)
            else:
                seg_dict = {
                    "text": getattr(item, "text", str(item)),
                    "start": getattr(item, "start_time", getattr(item, "start_offset", None)),
                    "end": getattr(item, "end_time", getattr(item, "end_offset", None)),
                }

            segments.append(
                {
                    "text": seg_dict.get("text"),
                    "start": self._extract_timestamp(seg_dict, "start"),
                    "end": self._extract_timestamp(seg_dict, "end"),
                }
            )
        return segments

    @staticmethod
    def _extract_timestamp(source: Dict[str, Any], prefix: str) -> Optional[float]:
        keys = (
            f"{prefix}_time",
            f"{prefix}_offset",
            prefix,
            f"{prefix}Time",
        )
        for key in keys:
            value = source.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None
