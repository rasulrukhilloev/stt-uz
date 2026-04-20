from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AudioWriteResult:
    output_path: Path
    sample_rate: int
    sample_count: int
    duration_seconds: float


def write_audio_to_wav(audio: np.ndarray, sample_rate: int, output_path: Path) -> AudioWriteResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return AudioWriteResult(
        output_path=output_path,
        sample_rate=sample_rate,
        sample_count=int(pcm.shape[0]),
        duration_seconds=float(pcm.shape[0] / sample_rate) if sample_rate else 0.0,
    )
