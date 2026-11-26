from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

SR = 16_000


def bursts(spans: list[tuple[float, float]], total: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    """Silence with tone bursts at the given (start, end) seconds -- a stand-in for speech."""
    pcm = np.zeros(int(total * SR), dtype=np.float32)
    for start, end in spans:
        idx = np.arange(int(start * SR), int(end * SR))
        pcm[idx] = amp * np.sin(2 * np.pi * freq * idx / SR)
    return pcm


def write_audio(path: Path, pcm: np.ndarray, fmt: str, codec: str, rate: int = SR, channels: int = 1) -> Path:
    """Encode 16 kHz float PCM into a real container/codec at another rate/layout."""
    layout = "mono" if channels == 1 else "stereo"
    s16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16).reshape(1, -1)
    with av.open(str(path), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=rate, layout=layout)
        resampler = av.AudioResampler(format=stream.codec_context.format, layout=layout, rate=rate)
        frame = av.AudioFrame.from_ndarray(s16, format="s16", layout="mono")
        frame.sample_rate = SR
        for f in resampler.resample(frame) + resampler.resample(None):
            out.mux(stream.encode(f))
        out.mux(stream.encode(None))
    return path


@pytest.fixture
def make_audio(tmp_path):
    def _make(pcm: np.ndarray, name: str = "a.wav", fmt: str = "wav", codec: str = "pcm_s16le", **kw) -> Path:
        return write_audio(tmp_path / name, pcm, fmt, codec, **kw)

    return _make
