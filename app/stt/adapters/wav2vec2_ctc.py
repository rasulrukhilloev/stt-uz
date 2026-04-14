from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.stt.base import TranscriptionOutput


class Wav2Vec2CtcAdapter:
    def __init__(
        self,
        model_id: str,
        language: str | None,
        device: str,
        cache_dir: Path,
        token: str | None,
        revision: str | None,
    ) -> None:
        self._model_id = model_id
        self._language = language
        self._device = torch.device(device)
        self._cache_dir = cache_dir
        self._token = token
        self._revision = revision
        self._processor: Wav2Vec2Processor | None = None
        self._model: Wav2Vec2ForCTC | None = None

    @property
    def model_name(self) -> str:
        return f"transformers:{self._model_id}"

    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def load(self) -> None:
        if self.is_loaded():
            return

        self._processor = Wav2Vec2Processor.from_pretrained(
            self._model_id,
            cache_dir=str(self._cache_dir),
            token=self._token,
            revision=self._revision,
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            self._model_id,
            cache_dir=str(self._cache_dir),
            token=self._token,
            revision=self._revision,
            use_safetensors=True,
        )
        self._model.to(self._device)
        self._model.eval()

    def unload(self) -> None:
        self._processor = None
        self._model = None

    def transcribe(self, audio_path: Path) -> TranscriptionOutput:
        if self._processor is None or self._model is None:
            raise RuntimeError("Model is not loaded")

        audio = _read_normalized_wav(audio_path)
        inputs = self._processor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self._device)
        attention_mask = None
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].to(self._device)

        with torch.inference_mode():
            logits = self._model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(predicted_ids)[0].strip()
        return TranscriptionOutput(text=text, language=self._language)


def _read_normalized_wav(audio_path: Path) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_pcm = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}")
    if channels != 1:
        raise ValueError(f"Expected mono WAV, got {channels} channels")
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz WAV, got {sample_rate} Hz")

    audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
    return audio / 32768.0
