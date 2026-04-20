from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.tts.base import SynthesisOutput, SynthesisRequest


class OmniVoiceAdapter:
    def __init__(
        self,
        model_id: str,
        language: str | None,
        device: str,
        dtype: str,
        cache_dir: Path,
        token: str | None,
        revision: str | None,
    ) -> None:
        self._model_id = model_id
        self._language = language
        self._device = device
        self._dtype = dtype
        self._cache_dir = cache_dir
        self._token = token
        self._revision = revision
        self._model = None

    @property
    def model_name(self) -> str:
        return f"omnivoice:{self._model_id}"

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded():
            return

        try:
            import torch
            from omnivoice import OmniVoice
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OmniVoice dependencies are not installed in the Python environment running the bot. "
                "Install them in the same interpreter, e.g. on Apple Silicon: "
                "`python -m pip install torch==2.8.0 torchaudio==2.8.0 && python -m pip install omnivoice`, "
                "then start the bot with that same `python`."
            ) from exc

        os.environ.setdefault("HF_HOME", str(self._cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(self._cache_dir))
        if self._token:
            os.environ.setdefault("HF_TOKEN", self._token)

        load_kwargs = {"device_map": self._device}
        resolved_dtype = _resolve_dtype(torch, self._dtype, self._device)
        if resolved_dtype is not None:
            load_kwargs["dtype"] = resolved_dtype

        self._model = OmniVoice.from_pretrained(self._model_id, **load_kwargs)

    def unload(self) -> None:
        self._model = None

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutput:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        mode = "auto"
        generate_kwargs = {
            "text": request.text,
        }
        language = request.language or self._language
        if language:
            generate_kwargs["language"] = language

        if request.reference_audio_path is not None:
            if request.reference_text is None and hasattr(self._model, "load_asr_model"):
                self._model.load_asr_model()
            generate_kwargs["ref_audio"] = str(request.reference_audio_path)
            if request.reference_text is not None:
                generate_kwargs["ref_text"] = request.reference_text
            mode = "voice-clone"
        elif request.instruct:
            generate_kwargs["instruct"] = request.instruct
            mode = "voice-design"

        audio = self._model.generate(**generate_kwargs)[0]
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        return SynthesisOutput(
            audio=waveform,
            sample_rate=24_000,
            mode=mode,
            language=language,
        )


def _resolve_dtype(torch_module, dtype_name: str, device: str):
    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        if device.startswith("cuda"):
            return torch_module.float16
        return torch_module.float32

    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    try:
        return dtype_map[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported TTS dtype: {dtype_name}") from exc
