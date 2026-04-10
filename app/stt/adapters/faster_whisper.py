from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from app.stt.base import TranscriptionOutput


class FasterWhisperAdapter:
    def __init__(
        self,
        model_id: str,
        language: str | None,
        device: str,
        compute_type: str,
        beam_size: int,
        download_root: Path,
    ) -> None:
        self._model_id = model_id
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._download_root = download_root
        self._model: WhisperModel | None = None

    @property
    def model_name(self) -> str:
        return f"faster-whisper:{self._model_id}"

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        self._model = WhisperModel(
            self._model_id,
            device=self._device,
            compute_type=self._compute_type,
            download_root=str(self._download_root),
        )

    def unload(self) -> None:
        self._model = None

    def transcribe(self, audio_path: Path) -> TranscriptionOutput:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=self._beam_size,
            language=self._language,
            task="transcribe",
        )
        materialized_segments = list(segments)
        text = "".join(segment.text for segment in materialized_segments).strip()
        return TranscriptionOutput(text=text, language=getattr(info, "language", self._language))

