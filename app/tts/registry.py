from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TtsModelSpec:
    model_id: str
    display_name: str
    family: str
    runtime: str
    revision: str | None = None


AVAILABLE_TTS_MODELS: tuple[TtsModelSpec, ...] = (
    TtsModelSpec(
        model_id="k2-fsa/OmniVoice",
        display_name="OmniVoice",
        family="omnilingual-diffusion",
        runtime="modal",
    ),
)


def get_available_tts_models() -> tuple[TtsModelSpec, ...]:
    return AVAILABLE_TTS_MODELS
