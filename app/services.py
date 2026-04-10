from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.storage.results import ResultsRepository
from app.stt.adapters.hf_whisper import HfWhisperAdapter
from app.stt.manager import ModelManager


@dataclass(slots=True)
class AppServices:
    settings: Settings
    results_repo: ResultsRepository
    model_manager: ModelManager

    @classmethod
    def build(cls, settings: Settings) -> "AppServices":
        results_repo = ResultsRepository(settings.sqlite_path)
        results_repo.init_db()

        adapter = HfWhisperAdapter(
            model_id=settings.stt_model_id,
            language=settings.stt_language,
            device=settings.stt_device,
            beam_size=settings.stt_beam_size,
            cache_dir=settings.model_cache_dir,
        )
        model_manager = ModelManager(adapter)
        return cls(settings=settings, results_repo=results_repo, model_manager=model_manager)
