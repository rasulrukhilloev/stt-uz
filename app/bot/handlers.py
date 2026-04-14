from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from time import perf_counter

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.helpers import escape_markdown
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.audio.normalize import AudioNormalizationResult, normalize_audio_to_wav
from app.services import AppServices
from app.storage.results import TranscriptionLogRecord

LOGGER = logging.getLogger(__name__)
MODEL_CALLBACK_PREFIX = "select_model:"


def register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("models", list_models_command))
    application.add_handler(CommandHandler("current", current_model_command))
    application.add_handler(CommandHandler("warmup", warmup_command))
    application.add_handler(CallbackQueryHandler(select_model_callback, pattern=f"^{MODEL_CALLBACK_PREFIX}"))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    services = get_services(context)
    await update.message.reply_text(
        "Send me a Telegram voice message and I will return the transcript, model name, "
        "and timing breakdown.\nUse /models to choose between available STT models.\n"
        "Use /help to see available commands.",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await update.message.reply_text(
        "\n".join(
            [
                "Available commands:",
                "/start - Intro and usage",
                "/help - Show command list",
                "/models - List models and choose active one",
                "/current - Show currently selected model",
                "/warmup - Download and load current model now",
            ]
        )
    )


async def list_models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    services = get_services(context)
    selected_model_id = get_selected_model_id(context, services)
    model_lines = []
    keyboard_rows = []

    for index, model in enumerate(services.model_manager.list_models(), start=1):
        model_link = f"https://huggingface.co/{model.model_id}"
        current_suffix = " current" if model.model_id == selected_model_id else ""
        model_lines.append(
            f"{index}. {escape_markdown(model.display_name, version=1)} "
            f"{escape_markdown(model.family, version=1)} "
            f"[link]({model_link})"
            f"{escape_markdown(current_suffix, version=1)}"
        )
        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    text=model.display_name,
                    callback_data=f"{MODEL_CALLBACK_PREFIX}{model.model_id}",
                )
            ]
        )

    await update.message.reply_text(
        "Available models:\n"
        + "\n".join(model_lines)
        + "\n\nTap a button to switch the active model.",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard_rows),
    )


async def current_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    services = get_services(context)
    selected_model_id = get_selected_model_id(context, services)
    model_spec = services.model_manager.get_model_spec(selected_model_id)
    await update.message.reply_text(
        f"Current model: {model_spec.display_name}\n`{model_spec.model_id}`",
        parse_mode="Markdown",
    )


async def warmup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    services = get_services(context)
    selected_model_id = get_selected_model_id(context, services)
    model_spec = services.model_manager.get_model_spec(selected_model_id)

    status_message = await update.message.reply_text(
        f"Warming up {model_spec.display_name}. This may take a while on first download/load."
    )

    try:
        warmup_result = await asyncio.to_thread(services.model_manager.warmup, selected_model_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to warm up model")
        await status_message.edit_text(
            "Warmup failed.\n"
            f"Model: {model_spec.display_name}\n"
            f"Error: {exc}"
        )
        return

    await status_message.edit_text(
        "\n".join(
            [
                f"Warmup complete for {warmup_result.model_display_name}",
                f"Model ID: {warmup_result.model_id}",
                f"Cold start: {'yes' if warmup_result.cold_start else 'no'}",
                f"Model load: {warmup_result.model_load_time_ms:.0f} ms",
            ]
        )
    )


async def select_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    services = get_services(context)
    selected_model_id = query.data.removeprefix(MODEL_CALLBACK_PREFIX)

    try:
        model_spec = services.model_manager.get_model_spec(selected_model_id)
    except ValueError:
        await query.answer("Unknown model", show_alert=True)
        return

    context.user_data["selected_model_id"] = selected_model_id
    await query.answer(f"Selected {model_spec.display_name}")
    await query.edit_message_text(
        f"Active model set to {model_spec.display_name}\n`{model_spec.model_id}`",
        parse_mode="Markdown",
    )


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.voice is None:
        return

    services = get_services(context)
    selected_model_id = get_selected_model_id(context, services)
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

            stt_result = await asyncio.to_thread(
                services.model_manager.transcribe,
                selected_model_id,
                normalized_path,
            )
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
                model_name=stt_result.model_id if status == "ok" else selected_model_id,
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
        model_spec = services.model_manager.get_model_spec(selected_model_id)
        await update.message.reply_text(
            "Transcription failed.\n"
            f"Model: {model_spec.display_name}\n"
            f"Total: {total_time_ms:.0f} ms\n"
            f"Error: {error_message}"
        )
        return

    await update.message.reply_text(
        "\n".join(
            [
                f"*Transcript:* {escape_markdown_v2(transcript_text or '[empty]')}",
                f"Model: `{escape_markdown_v2(stt_result.model_display_name)}`",
                f"Model ID: `{escape_markdown_v2(stt_result.model_id)}`",
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


def get_selected_model_id(context: ContextTypes.DEFAULT_TYPE, services: AppServices) -> str:
    selected_model_id = context.user_data.get("selected_model_id", services.settings.stt_model_id)
    try:
        services.model_manager.get_model_spec(selected_model_id)
    except ValueError:
        selected_model_id = services.settings.stt_model_id
        context.user_data["selected_model_id"] = selected_model_id
    return selected_model_id


def elapsed_ms(start_time: float) -> float:
    return (perf_counter() - start_time) * 1000.0


def escape_markdown_v2(value: str) -> str:
    special_chars = r"_*[]()~`>#+-=|{}.!"
    escaped = value
    for char in special_chars:
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def get_bot_commands() -> list[BotCommand]:
    return [
        BotCommand("start", "Intro and usage"),
        BotCommand("help", "Show available commands"),
        BotCommand("models", "List and select STT models"),
        BotCommand("current", "Show current model"),
        BotCommand("warmup", "Download and load current model"),
    ]
