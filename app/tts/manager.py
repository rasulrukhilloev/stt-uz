from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from app.tts.base import SynthesisRequest, TtsAdapter
from app.tts.registry import TtsModelSpec


@dataclass(slots=True)
class ManagedSynthesisResult:
    audio: np.ndarray
    sample_rate: int
    mode: str
    language: str | None
    model_id: str
    model_display_name: str
    cold_start: bool
    model_load_time_ms: float
    inference_time_ms: float


@dataclass(slots=True)
class TtsWarmupResult:
    model_id: str
    model_display_name: str
    cold_start: bool
    model_load_time_ms: float


class TtsManager:
    def __init__(
        self,
        model_specs: tuple[TtsModelSpec, ...],
        adapter_factory: Callable[[str], TtsAdapter],
    ) -> None:
        self._model_specs = {spec.model_id: spec for spec in model_specs}
        self._adapter_factory = adapter_factory
        self._operation_lock = threading.Lock()
        self._active_model_id: str | None = None
        self._active_adapter: TtsAdapter | None = None

    @property
    def default_model_id(self) -> str:
        return next(iter(self._model_specs))

    def list_models(self) -> tuple[TtsModelSpec, ...]:
        return tuple(self._model_specs.values())

    def get_model_spec(self, model_id: str) -> TtsModelSpec:
        try:
            return self._model_specs[model_id]
        except KeyError as exc:
            raise ValueError(f"Unsupported TTS model: {model_id}") from exc

    def synthesize(self, model_id: str, request: SynthesisRequest) -> ManagedSynthesisResult:
        model_spec = self.get_model_spec(model_id)

        with self._operation_lock:
            cold_start = False
            model_load_time_ms = 0.0
            adapter = self._ensure_active_adapter(model_id)

            if not adapter.is_loaded():
                cold_start = True
                load_started = perf_counter()
                adapter.load()
                model_load_time_ms = elapsed_ms(load_started)

            inference_started = perf_counter()
            output = adapter.synthesize(request)
            inference_time_ms = elapsed_ms(inference_started)

        return ManagedSynthesisResult(
            audio=output.audio,
            sample_rate=output.sample_rate,
            mode=output.mode,
            language=output.language,
            model_id=model_spec.model_id,
            model_display_name=model_spec.display_name,
            cold_start=cold_start,
            model_load_time_ms=model_load_time_ms,
            inference_time_ms=inference_time_ms,
        )

    def warmup(self, model_id: str) -> TtsWarmupResult:
        model_spec = self.get_model_spec(model_id)

        with self._operation_lock:
            cold_start = False
            model_load_time_ms = 0.0
            adapter = self._ensure_active_adapter(model_id)

            if not adapter.is_loaded():
                cold_start = True
                load_started = perf_counter()
                adapter.load()
                model_load_time_ms = elapsed_ms(load_started)

        return TtsWarmupResult(
            model_id=model_spec.model_id,
            model_display_name=model_spec.display_name,
            cold_start=cold_start,
            model_load_time_ms=model_load_time_ms,
        )

    def _ensure_active_adapter(self, model_id: str) -> TtsAdapter:
        if self._active_model_id == model_id and self._active_adapter is not None:
            return self._active_adapter

        if self._active_adapter is not None:
            self._active_adapter.unload()

        self._active_adapter = self._adapter_factory(model_id)
        self._active_model_id = model_id
        return self._active_adapter


def elapsed_ms(start_time: float) -> float:
    return (perf_counter() - start_time) * 1000.0
