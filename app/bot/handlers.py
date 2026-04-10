from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from time import perf_counter

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.audio.normalize import AudioNormalizationResult, normalize_audio_to_wav
from app.services import AppServices
from app.storage.results import TranscriptionLogRecord

LOGGER = logging.getLogger(__name__)


def register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    services = get_services(context)
    await update.message.reply_text(
        "Send me a Telegram voice message and I will return the transcript, "
        f"model name, and timing breakdown for `{services.model_manager.model_name}`.",
        parse_mode="Markdown",
    )


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.voice is None:
        return

    services = get_services(context)
    voice = update.message.voice
    total_start = perf_counter()
    temp_dir = services.settings.temp_audio_dir

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        original_path = tmp_dir / f"{voice.file_unique_id}.ogg"
        normalized_path = tmp_dir / f"{voice.file_unique_id}.wav"

        download_time_ms = 0.0
        preprocess_time_ms = 0.0
        load_time_ms = 0.0
        inference_time_ms = 0.0
        actual_duration_seconds = None
        transcript_text = ""
        cold_start = False
        status = "ok"
        error_message = None

        try:
            download_started = perf_counter()
            telegram_file = await context.bot.get_file(voice.file_id)
            await telegram_file.download_to_drive(custom_path=str(original_path))
            download_time_ms = elapsed_ms(download_started)

            preprocess_started = perf_counter()
            normalization_result = await asyncio.to_thread(
                normalize_audio_to_wav,
                original_path,
                normalized_path,
            )
            preprocess_time_ms = elapsed_ms(preprocess_started)
            actual_duration_seconds = normalization_result.duration_seconds

            stt_result = await asyncio.to_thread(services.model_manager.transcribe, normalized_path)
            load_time_ms = stt_result.model_load_time_ms
            inference_time_ms = stt_result.inference_time_ms
            cold_start = stt_result.cold_start
            transcript_text = stt_result.text
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_message = str(exc)
            LOGGER.exception("Failed to transcribe Telegram voice message")
        finally:
            total_time_ms = elapsed_ms(total_start)
            record = TranscriptionLogRecord(
                telegram_file_id=voice.file_id,
                telegram_duration_seconds=voice.duration,
                normalized_duration_seconds=actual_duration_seconds,
                model_name=services.model_manager.model_name,
                language=services.settings.stt_language,
                cold_start=cold_start,
                download_time_ms=download_time_ms,
                preprocess_time_ms=preprocess_time_ms,
                model_load_time_ms=load_time_ms,
                inference_time_ms=inference_time_ms,
                total_time_ms=total_time_ms,
                transcript=transcript_text,
                status=status,
                error_message=error_message,
            )
            await asyncio.to_thread(services.results_repo.insert_log, record)

    if status == "error":
        await update.message.reply_text(
            "Transcription failed.\n"
            f"Model: {services.model_manager.model_name}\n"
            f"Total: {total_time_ms:.0f} ms\n"
            f"Error: {error_message}"
        )
        return

    await update.message.reply_text(
        "\n".join(
            [
                f"*Transcript:* {escape_markdown_v2(transcript_text or '[empty]')}",
                f"Model: `{escape_markdown_v2(services.model_manager.model_name)}`",
                f"Cold start: {'yes' if cold_start else 'no'}",
                f"Preprocess: {preprocess_time_ms:.0f} ms",
                f"Model load: {load_time_ms:.0f} ms",
                f"Inference: {inference_time_ms:.0f} ms",
                f"Total: {total_time_ms:.0f} ms",
            ]
        ),
        parse_mode="MarkdownV2",
    )


def get_services(context: ContextTypes.DEFAULT_TYPE) -> AppServices:
    return context.application.bot_data["services"]


def elapsed_ms(start_time: float) -> float:
    return (perf_counter() - start_time) * 1000.0


def escape_markdown_v2(value: str) -> str:
    special_chars = r"_*[]()~`>#+-=|{}.!"
    escaped = value
    for char in special_chars:
        escaped = escaped.replace(char, f"\\{char}")
    return escaped
