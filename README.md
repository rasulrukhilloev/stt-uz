# Uzbek Speech Telegram Bot Prototype

Phase 1 prototype for evaluating self-hosted Uzbek STT and TTS models through Telegram.

## What it does

- Runs a local Telegram bot with long polling
- Accepts Telegram voice messages
- Lets the user choose between a small set of STT models
- Adds a TTS path powered by `k2-fsa/OmniVoice`
- Normalizes audio to `16 kHz` mono WAV
- Transcribes audio with Hugging Face speech checkpoints
- Synthesizes speech from `/say <text>` and can clone a replied-to voice note
- Replies with transcripts or generated audio plus timing breakdown and cold/warm status
- Saves STT and TTS runs to SQLite for later comparison

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

- Base install is `pip install -e .`.
- OmniVoice is optional: `pip install -e '.[tts]'`.
- OmniVoice is not installable on Intel Mac `x86_64` because upstream PyTorch stopped publishing macOS x86 wheels after `2.2.x`.
- STT uses `/models`, `/current`, and `/warmup`.
- TTS uses `/ttsmodels`, `/ttscurrent`, `/ttswarmup`, and `/say`.
- `/say <voice prompt> | <text>` uses OmniVoice voice design mode.
- Reply to a Telegram voice note with `/say <text>` to use OmniVoice voice cloning.
- On supported platforms, the TTS extra will upgrade `torch`/`transformers` as needed for OmniVoice.
- The model is downloaded and cached locally on first load.
- The cache path is controlled by `MODEL_CACHE_DIR`.
- `inference_time` excludes model loading time.
- `total_time` includes the full request lifecycle.
