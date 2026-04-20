from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import modal
from pydantic import BaseModel

app = modal.App("omnivoice-tts")

MODEL_ID = "k2-fsa/OmniVoice"
CACHE_DIR = Path("/model-cache")

model_volume = modal.Volume.from_name("omnivoice-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("omnivoice>=0.1.3", "numpy>=1.26.0", "pydantic>=2.0")
)


class SynthesisPayload(BaseModel):
    text: str
    language: str | None = None
    instruct: str | None = None
    reference_audio_b64: str | None = None
    reference_text: str | None = None


@app.cls(
    gpu="T4",
    image=image,
    volumes={CACHE_DIR: model_volume},
    scaledown_window=300,
    timeout=120,
)
class OmniVoiceEndpoint:
    @modal.build()
    def download_model(self) -> None:
        os.environ["HF_HOME"] = str(CACHE_DIR)
        from omnivoice import OmniVoice  # noqa: PLC0415

        OmniVoice.from_pretrained(MODEL_ID)

    @modal.enter()
    def load_model(self) -> None:
        import torch  # noqa: PLC0415
        from omnivoice import OmniVoice  # noqa: PLC0415

        os.environ["HF_HOME"] = str(CACHE_DIR)
        self.model = OmniVoice.from_pretrained(
            MODEL_ID, device_map="cuda", dtype=torch.float16
        )

    @modal.web_endpoint(method="POST")
    def synthesize(self, payload: SynthesisPayload) -> dict:
        import numpy as np  # noqa: PLC0415

        generate_kwargs: dict = {"text": payload.text}
        mode = "auto"

        if payload.language:
            generate_kwargs["language"] = payload.language

        ref_tmp = None
        try:
            if payload.reference_audio_b64 is not None:
                audio_bytes = base64.b64decode(payload.reference_audio_b64)
                ref_tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
                ref_tmp.write(audio_bytes)
                ref_tmp.flush()
                generate_kwargs["ref_audio"] = ref_tmp.name
                if payload.reference_text:
                    generate_kwargs["ref_text"] = payload.reference_text
                mode = "voice-clone"
            elif payload.instruct:
                generate_kwargs["instruct"] = payload.instruct
                mode = "voice-design"

            audio = self.model.generate(**generate_kwargs)[0]
        finally:
            if ref_tmp is not None:
                ref_tmp.close()
                Path(ref_tmp.name).unlink(missing_ok=True)

        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_b64 = base64.b64encode(waveform.tobytes()).decode()

        return {"audio_b64": audio_b64, "sample_rate": 24_000, "mode": mode}
