"""Silence-aware chunking for long audio.

Why chunk at all when Whisper already windows internally at 30 s?
  * bounded memory: we never hold more than ~one chunk of PCM;
  * fault isolation + retry: a failure costs one chunk, not a 2-hour job;
  * progress reporting, and a natural unit for fan-out to multiple workers;
  * hosted STT APIs have hard per-request size/duration limits.

Cutting blindly at N seconds splits words, which both engines and humans
transcribe badly. Instead, as the buffer reaches the target length we look
back `cut_search_seconds` for the quietest 30 ms frame and cut there. If even
the quietest frame is loud (continuous speech/music), we hard-cut and start
the next chunk `overlap_seconds` earlier. Each side of such a seam owns only
the half of the overlap nearest its own centre -- the half where it had more
acoustic context -- via `[keep_from, keep_until)`. Ownership is decided per
*word* by the merger: segment-level ownership duplicated words (measured).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from transcriber.audio import SAMPLE_RATE

FRAME = int(0.030 * SAMPLE_RATE)  # 30 ms analysis frame


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float            # absolute offset of samples[0] in the source, seconds
    keep_from: float        # absolute time window this chunk owns: words centred in
    keep_until: float       #   [keep_from, keep_until) are kept, the rest belong to a neighbour
    samples: np.ndarray
    clean_cut: bool         # True if this chunk ended on silence

    def owns(self, t: float) -> bool:
        return self.keep_from <= t < self.keep_until

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.start + self.duration


def find_cut(samples: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    """Index (in samples) of the quietest frame centre in samples[lo:hi], and its RMS."""
    window = samples[lo:hi]
    n_frames = len(window) // FRAME
    if n_frames == 0:
        return hi, float("inf")
    frames = window[: n_frames * FRAME].reshape(n_frames, FRAME)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    # Prefer the latest of equally-quiet frames so chunks stay close to target length.
    best = n_frames - 1 - int(np.argmin(rms[::-1]))
    return lo + best * FRAME + FRAME // 2, float(rms[best])


def active_ratio(samples: np.ndarray, silence_rms: float) -> float:
    """Fraction of 30 ms frames louder than `silence_rms`: how much of the chunk has *any* sound."""
    n = len(samples) // FRAME
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(samples[: n * FRAME].reshape(n, FRAME) ** 2, axis=1))
    return float(np.mean(rms > silence_rms))


def chunk_stream(
    blocks: Iterable[np.ndarray],
    chunk_seconds: float = 120.0,
    cut_search_seconds: float = 10.0,
    silence_rms: float = 0.01,
    overlap_seconds: float = 1.0,
) -> Iterator[Chunk]:
    target = int(chunk_seconds * SAMPLE_RATE)
    search = min(int(cut_search_seconds * SAMPLE_RATE), target // 2)
    overlap = int(overlap_seconds * SAMPLE_RATE)

    pending: list[np.ndarray] = []
    pending_len = 0
    buf_start = 0        # absolute sample index of pending[0][0]
    keep_from = 0        # absolute sample index
    index = 0

    for block in blocks:
        pending.append(block)
        pending_len += len(block)
        while pending_len >= target:
            buf = np.concatenate(pending)
            cut, rms = find_cut(buf, target - search, target)
            clean = rms <= silence_rms
            # On a hard cut, re-feed `overlap` samples so the next chunk has left context,
            # and split ownership of the overlap down its middle.
            next_start = cut if clean else max(cut - overlap, 0)
            boundary = buf_start + (cut if clean else cut - (cut - next_start) // 2)
            yield Chunk(index, buf_start / SAMPLE_RATE, keep_from / SAMPLE_RATE,
                        boundary / SAMPLE_RATE, buf[:cut], clean)
            index += 1
            keep_from = boundary
            buf_start += next_start
            rest = buf[next_start:]
            pending, pending_len = [rest], len(rest)

    if pending_len:
        buf = np.concatenate(pending)
        # A tail shorter than one frame carries nothing a model could transcribe.
        if index == 0 or len(buf) - (keep_from - buf_start) >= FRAME:
            yield Chunk(index, buf_start / SAMPLE_RATE, keep_from / SAMPLE_RATE, float("inf"), buf, True)
