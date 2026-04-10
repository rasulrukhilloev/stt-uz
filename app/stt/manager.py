from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.stt.base import SttAdapter


@dataclass(slots=True)
class ManagedTranscriptionResult:
    text: str
    language: str | None
    cold_start: bool
    model_load_time_ms: float
    inference_time_ms: float


class ModelManager:
    def __init__(self, adapter: SttAdapter) -> None:
        self._adapter = adapter
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._adapter.model_name

    def transcribe(self, audio_path: Path) -> ManagedTranscriptionResult:
        cold_start = False
        model_load_time_ms = 0.0

        if not self._adapter.is_loaded():
            with self._load_lock:
                if not self._adapter.is_loaded():
                    cold_start = True
                    load_started = perf_counter()
                    self._adapter.load()
                    model_load_time_ms = elapsed_ms(load_started)

        inference_started = perf_counter()
        output = self._adapter.transcribe(audio_path)
        inference_time_ms = elapsed_ms(inference_started)

        return ManagedTranscriptionResult(
            text=output.text,
            language=output.language,
            cold_start=cold_start,
            model_load_time_ms=model_load_time_ms,
            inference_time_ms=inference_time_ms,
        )


def elapsed_ms(start_time: float) -> float:
    return (perf_counter() - start_time) * 1000.0
