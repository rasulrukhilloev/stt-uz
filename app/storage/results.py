from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.storage.db import connect


@dataclass(slots=True)
class TranscriptionLogRecord:
    telegram_file_id: str
    telegram_duration_seconds: int | None
    normalized_duration_seconds: float | None
    model_name: str
    language: str | None
    cold_start: bool
    download_time_ms: float
    preprocess_time_ms: float
    model_load_time_ms: float
    inference_time_ms: float
    total_time_ms: float
    transcript: str
    status: str
    error_message: str | None


class ResultsRepository:
    def __init__(self, sqlite_path: Path) -> None:
        self._sqlite_path = sqlite_path

    def init_db(self) -> None:
        with connect(self._sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transcription_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    telegram_file_id TEXT NOT NULL,
                    telegram_duration_seconds INTEGER,
                    normalized_duration_seconds REAL,
                    model_name TEXT NOT NULL,
                    language TEXT,
                    cold_start INTEGER NOT NULL,
                    download_time_ms REAL NOT NULL,
                    preprocess_time_ms REAL NOT NULL,
                    model_load_time_ms REAL NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    total_time_ms REAL NOT NULL,
                    transcript TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )

    def insert_log(self, record: TranscriptionLogRecord) -> None:
        with connect(self._sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO transcription_logs (
                    created_at,
                    telegram_file_id,
                    telegram_duration_seconds,
                    normalized_duration_seconds,
                    model_name,
                    language,
                    cold_start,
                    download_time_ms,
                    preprocess_time_ms,
                    model_load_time_ms,
                    inference_time_ms,
                    total_time_ms,
                    transcript,
                    status,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    record.telegram_file_id,
                    record.telegram_duration_seconds,
                    record.normalized_duration_seconds,
                    record.model_name,
                    record.language,
                    int(record.cold_start),
                    record.download_time_ms,
                    record.preprocess_time_ms,
                    record.model_load_time_ms,
                    record.inference_time_ms,
                    record.total_time_ms,
                    record.transcript,
                    record.status,
                    record.error_message,
                ),
            )
