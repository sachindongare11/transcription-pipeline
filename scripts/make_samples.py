"""Generate mock speech in several formats, plus a multi-minute "long" file.

Uses macOS `say` for real synthetic speech (so Whisper has something to
transcribe) and PyAV to re-encode it, so we exercise genuinely different
containers/codecs/sample rates rather than renamed WAVs.

    python scripts/make_samples.py [out_dir]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import av
import numpy as np

SCRIPT = [
    "Good morning everyone, and welcome to the quarterly regulatory review.",
    "Today we will cover the new incident reporting obligations under DORA.",
    "Major ICT related incidents must be reported to the competent authority.",
    "The initial notification is due within four hours of classification.",
    "An intermediate report follows within seventy two hours.",
    "Finally, a root cause analysis is due within one month.",
    "Please send any questions to the compliance team after this call.",
]

# (filename, container format, codec, sample rate, channels)
VARIANTS = [
    ("sample.wav", "wav", "pcm_s16le", 22050, 1),
    ("sample.mp3", "mp3", "libmp3lame", 44100, 2),
    ("sample.flac", "flac", "flac", 48000, 2),
    ("sample.m4a", "ipod", "aac", 44100, 2),
    ("sample.ogg", "ogg", "libopus", 48000, 1),
]


def say(text: str, rate: int = 22050) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "say.wav"
        subprocess.run(["say", "-o", str(out), f"--data-format=LEI16@{rate}", text], check=True)
        with av.open(str(out)) as c:
            return np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(audio=0)]).astype(np.int16)


def write(path: Path, pcm: np.ndarray, src_rate: int, fmt: str, codec: str, rate: int, channels: int) -> None:
    layout = "mono" if channels == 1 else "stereo"
    with av.open(str(path), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=rate, layout=layout)
        resampler = av.AudioResampler(format=stream.codec_context.format, layout=layout, rate=rate)
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = src_rate
        for f in resampler.resample(frame) + resampler.resample(None):
            out.mux(stream.encode(f))
        out.mux(stream.encode(None))


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = 22050

    short = say(" ".join(SCRIPT[:3]), rate)
    for name, fmt, codec, sr, ch in VARIANTS:
        write(out_dir / name, short, rate, fmt, codec, sr, ch)
        print("wrote", out_dir / name)

    # ~5 minutes: the full script repeated, with pauses of varying length and a
    # stretch of low-level noise so chunk cuts land in realistic places.
    rng = np.random.default_rng(0)
    parts = []
    for rep in range(6):
        for line in SCRIPT:
            parts.append(say(f"Part {rep + 1}. {line}", rate))
            parts.append(np.zeros(int(rate * rng.uniform(0.4, 2.5)), dtype=np.int16))
        parts.append((rng.normal(0, 60, rate * 4)).astype(np.int16))  # 4 s of room noise
    write(out_dir / "long.mp3", np.concatenate(parts), rate, "mp3", "libmp3lame", 44100, 1)
    print("wrote", out_dir / "long.mp3")

    (out_dir / "not_audio.mp3").write_text("this is a text file wearing an .mp3 extension\n")
    print("wrote", out_dir / "not_audio.mp3")


if __name__ == "__main__":
    main()
