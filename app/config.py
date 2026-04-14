from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    telegram_bot_token: str
    huggingface_token: str | None
    stt_model_id: str
    stt_language: str | None
    stt_device: str
    stt_compute_type: str
    stt_beam_size: int
    model_cache_dir: Path
    sqlite_path: Path
    temp_audio_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "data/model_cache"))
        sqlite_path = Path(os.getenv("SQLITE_PATH", "data/results.sqlite3"))
        temp_audio_dir = Path(os.getenv("TEMP_AUDIO_DIR", "data/temp_audio"))

        model_cache_dir.mkdir(parents=True, exist_ok=True)
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        temp_audio_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            telegram_bot_token=token,
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN", "").strip() or None,
            stt_model_id=os.getenv("STT_MODEL_ID", "small"),
            stt_language=os.getenv("STT_LANGUAGE", "uz") or None,
            stt_device=os.getenv("STT_DEVICE", "cpu"),
            stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "int8"),
            stt_beam_size=int(os.getenv("STT_BEAM_SIZE", "1")),
            model_cache_dir=model_cache_dir,
            sqlite_path=sqlite_path,
            temp_audio_dir=temp_audio_dir,
        )
