from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    display_name: str
    family: str
    runtime: str
    revision: str | None = None


AVAILABLE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="Kotib/uzbek_stt_v1",
        display_name="Kotib Uzbek STT v1",
        family="whisper-medium",
        runtime="hf-whisper",
    ),
    ModelSpec(
        model_id="islomov/rubaistt_v2_medium",
        display_name="rubaiSTT v2 Medium",
        family="whisper-medium",
        runtime="hf-whisper",
    ),
    ModelSpec(
        model_id="Bahrom1996/stt_uzbek_medium_2025",
        display_name="Bahrom STT Uzbek Medium 2025",
        family="whisper-medium",
        runtime="hf-whisper",
    ),
)


def get_available_models() -> tuple[ModelSpec, ...]:
    return AVAILABLE_MODELS
