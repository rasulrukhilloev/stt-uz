# Uzbek STT Telegram Bot Prototype

Phase 1 prototype for evaluating a single self-hosted Uzbek STT model through Telegram.

## What it does

- Runs a local Telegram bot with long polling
- Accepts Telegram voice messages
- Lets the user choose between a small set of STT models
- Normalizes audio to `16 kHz` mono WAV
- Transcribes audio with a Hugging Face Whisper checkpoint
- Replies with transcript, model name, timing breakdown, and cold/warm status
- Saves each run to SQLite for later comparison

## Why this design

This version keeps the architecture intentionally small:

- one bot process
- one model in memory
- lazy model loading
- SQLite instead of a heavier database
- no webhook, no ngrok, no frontend

The goal is to measure the pipeline correctly before adding more models.

## Setup

1. Create a virtualenv and install dependencies.
2. Copy `.env.example` to `.env` and fill in the Telegram bot token.
3. Run:

```bash
python -m app.main
```

## Notes

- The model is downloaded and cached locally on first load.
- The cache path is controlled by `MODEL_CACHE_DIR`.
- `inference_time` excludes model loading time.
- `total_time` includes the full request lifecycle.
