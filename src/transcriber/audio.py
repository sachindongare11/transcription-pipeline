"""Format-agnostic audio ingest.

Every input (WAV, MP3, M4A/AAC, FLAC, OGG/Opus, WebM, even video containers)
is decoded with PyAV -- libav/FFmpeg bundled in the wheel, so no system ffmpeg
binary -- and normalised to the single representation the rest of the
pipeline understands: 16 kHz, mono, float32 PCM in [-1, 1].

Decoding is a generator of small blocks so a multi-hour file never has to sit
in memory as a whole; the chunker downstream bounds what is buffered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np

SAMPLE_RATE = 16_000  # what Whisper-family models are trained on


class AudioDecodeError(ValueError):
    """The file is not decodable audio (corrupt, unsupported, or has no audio stream)."""


@dataclass(frozen=True)
class AudioInfo:
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration: float | None  # header-declared; may be absent or wrong for some streams


def probe(path: str | Path) -> AudioInfo:
    """Inspect the container header. We trust content, not the file extension."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioDecodeError("file contains no audio stream")
            stream = container.streams.audio[0]
            duration = None
            if container.duration is not None:
                duration = container.duration / av.time_base
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            return AudioInfo(
                container=container.format.name,
                codec=stream.codec_context.name,
                sample_rate=stream.codec_context.sample_rate,
                channels=stream.codec_context.channels,
                duration=duration,
            )
    except av.FFmpegError as exc:
        raise AudioDecodeError(f"cannot open audio: {_reason(exc)}") from exc


def stream_pcm(path: str | Path) -> Iterator[np.ndarray]:
    """Yield 16 kHz mono float32 blocks, in order, for the first audio stream."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            for frame in container.decode(stream):
                frame.pts = None  # let the resampler ignore broken/discontinuous timestamps
                for out in resampler.resample(frame):
                    yield _to_float(out)
            for out in resampler.resample(None):  # flush resampler tail
                yield _to_float(out)
    except av.FFmpegError as exc:
        raise AudioDecodeError(f"cannot decode audio: {_reason(exc)}") from exc


def load_pcm(path: str | Path) -> np.ndarray:
    """Whole file as one array -- convenience for short clips and tests."""
    blocks = list(stream_pcm(path))
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def _reason(exc: av.FFmpegError) -> str:
    # str(exc) embeds the file path -- on the server, an internal temp path. Callers
    # surface this message to clients, so keep only FFmpeg's reason.
    return exc.strerror or type(exc).__name__


def _to_float(frame: av.AudioFrame) -> np.ndarray:
    return frame.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
