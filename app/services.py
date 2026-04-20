from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.storage.results import ResultsRepository
from app.stt.adapters.hf_whisper import HfWhisperAdapter
from app.stt.adapters.wav2vec2_ctc import Wav2Vec2CtcAdapter
from app.stt.manager import ModelManager
from app.stt.registry import ModelSpec, get_available_models
from app.tts.adapters.modal_tts import ModalTtsAdapter
from app.tts.manager import TtsManager
from app.tts.registry import TtsModelSpec, get_available_tts_models


@dataclass(slots=True)
class AppServices:
    settings: Settings
    results_repo: ResultsRepository
    model_manager: ModelManager
    tts_manager: TtsManager

    @classmethod
    def build(cls, settings: Settings) -> "AppServices":
        results_repo = ResultsRepository(settings.sqlite_path)
        results_repo.init_db()

        model_manager = ModelManager(
            model_specs=get_available_models(),
            adapter_factory=lambda model_id: build_stt_adapter(
                model_id=model_id,
                settings=settings,
                model_specs=get_available_models(),
            ),
        )
        tts_manager = TtsManager(
            model_specs=get_available_tts_models(),
            adapter_factory=lambda model_id: build_tts_adapter(
                model_id=model_id,
                settings=settings,
                model_specs=get_available_tts_models(),
            ),
        )
        return cls(
            settings=settings,
            results_repo=results_repo,
            model_manager=model_manager,
            tts_manager=tts_manager,
        )


def build_stt_adapter(model_id: str, settings: Settings, model_specs: tuple[ModelSpec, ...]):
    model_spec = next((spec for spec in model_specs if spec.model_id == model_id), None)
    if model_spec is None:
        raise ValueError(f"Unsupported model: {model_id}")

    if model_spec.runtime == "hf-whisper":
        return HfWhisperAdapter(
            model_id=model_id,
            language=settings.stt_language,
            device=settings.stt_device,
            beam_size=settings.stt_beam_size,
            cache_dir=settings.model_cache_dir,
            token=settings.huggingface_token,
            revision=model_spec.revision,
        )

    if model_spec.runtime == "wav2vec2-ctc":
        return Wav2Vec2CtcAdapter(
            model_id=model_id,
            language=settings.stt_language,
            device=settings.stt_device,
            cache_dir=settings.model_cache_dir,
            token=settings.huggingface_token,
            revision=model_spec.revision,
        )

    raise ValueError(f"Unsupported model runtime: {model_spec.runtime}")


def build_tts_adapter(model_id: str, settings: Settings, model_specs: tuple[TtsModelSpec, ...]):
    model_spec = next((spec for spec in model_specs if spec.model_id == model_id), None)
    if model_spec is None:
        raise ValueError(f"Unsupported TTS model: {model_id}")

    if model_spec.runtime == "modal":
        if not settings.modal_tts_url:
            raise ValueError("MODAL_TTS_URL must be set to use the Modal TTS endpoint")
        return ModalTtsAdapter(
            endpoint_url=settings.modal_tts_url,
            language=settings.tts_language,
        )

    raise ValueError(f"Unsupported TTS model runtime: {model_spec.runtime}")
