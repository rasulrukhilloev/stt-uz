from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


@dataclass(slots=True)
class AudioNormalizationResult:
    output_path: Path
    sample_rate: int
    channels: int
    sample_count: int
    duration_seconds: float


def normalize_audio_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> AudioNormalizationResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    sample_count = 0

    with av.open(str(input_path)) as container, wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(TARGET_CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio stream found in the Telegram file")

        for frame in container.decode(audio=audio_stream.index):
            sample_count += _write_resampled_frames(wav_file, resampler.resample(frame))

        sample_count += _write_resampled_frames(wav_file, resampler.resample(None))

    return AudioNormalizationResult(
        output_path=output_path,
        sample_rate=sample_rate,
        channels=TARGET_CHANNELS,
        sample_count=sample_count,
        duration_seconds=sample_count / sample_rate if sample_rate else 0.0,
    )


def _write_resampled_frames(wav_file: wave.Wave_write, frames: object) -> int:
    if frames is None:
        return 0

    if not isinstance(frames, list):
        frames = [frames]

    written_samples = 0
    for frame in frames:
        pcm = np.asarray(frame.to_ndarray()).reshape(-1).astype(np.int16, copy=False)
        wav_file.writeframes(pcm.tobytes())
        written_samples += int(pcm.shape[0])
    return written_samples

