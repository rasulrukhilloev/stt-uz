from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class TranscriptionOutput:
    text: str
    language: str | None


class SttAdapter(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def is_loaded(self) -> bool:
        ...

    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...

    def transcribe(self, audio_path: Path) -> TranscriptionOutput:
        ...

