from __future__ import annotations

import base64
from pathlib import Path

import httpx
import numpy as np

from app.tts.base import SynthesisOutput, SynthesisRequest


class ModalTtsAdapter:
    def __init__(
        self,
        endpoint_url: str,
        language: str | None,
        timeout: float = 120.0,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._language = language
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return "modal:omnivoice"

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutput:
        language = request.language or self._language
        payload: dict = {
            "text": request.text,
            "language": language,
            "instruct": request.instruct,
            "reference_audio_b64": None,
            "reference_text": request.reference_text,
        }

        if request.reference_audio_path is not None:
            raw = Path(request.reference_audio_path).read_bytes()
            payload["reference_audio_b64"] = base64.b64encode(raw).decode()

        response = httpx.post(
            self._endpoint_url,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        audio_bytes = base64.b64decode(data["audio_b64"])
        audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()

        return SynthesisOutput(
            audio=audio,
            sample_rate=data["sample_rate"],
            mode=data["mode"],
            language=language,
        )
