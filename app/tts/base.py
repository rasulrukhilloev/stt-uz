from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class SynthesisRequest:
    text: str
    language: str | None
    instruct: str | None = None
    reference_audio_path: Path | None = None
    reference_text: str | None = None


@dataclass(slots=True)
class SynthesisOutput:
    audio: np.ndarray
    sample_rate: int
    mode: str
    language: str | None


class TtsAdapter(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def is_loaded(self) -> bool:
        ...

    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutput:
        ...
