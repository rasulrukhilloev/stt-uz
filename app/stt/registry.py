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
        model_id="jmshd/whisper-uz",
        display_name="jmshd Whisper UZ",
        family="whisper-base",
        runtime="hf-whisper",
    ),
    ModelSpec(
        model_id="aisha-org/Whisper-Uzbek",
        display_name="AISHA Whisper Uzbek",
        family="whisper-medium",
        runtime="hf-whisper",
        revision="b5080285102e27944b91324771f2521eccd6abe2",
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
    ModelSpec(
        model_id="oyqiz/uzbek_stt",
        display_name="Oyqiz Uzbek STT",
        family="wav2vec2-ctc",
        runtime="wav2vec2-ctc",
        revision="2cbbc37092c02b03d13c9015e0e90f430acb6c52",
    ),
    ModelSpec(
        model_id="sarahai/uzbek-stt-3",
        display_name="SarahAI Uzbek STT 3",
        family="wav2vec2-ctc",
        runtime="wav2vec2-ctc",
    ),
    ModelSpec(
        model_id="oyqiz/uzbek_stt_5_version",
        display_name="Oyqiz Uzbek STT v5",
        family="wav2vec2-ctc",
        runtime="wav2vec2-ctc",
        revision="76ddc6817f9aabda7a1d047c6930dc3450f2d713",
    ),
)


def get_available_models() -> tuple[ModelSpec, ...]:
    return AVAILABLE_MODELS
