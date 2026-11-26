"""Turn raw engine output into clean, downstream-ready segments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from transcriber.schemas import Segment, Word

_WS = re.compile(r"\s+")
LOW_CONFIDENCE = 0.5


@dataclass
class TimedWord:
    start: float
    end: float
    text: str
    probability: float


@dataclass
class TimedSegment:
    """An engine segment already shifted onto the source timeline."""

    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[TimedWord] = field(default_factory=list)


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip()


# Stock phrases Whisper emits over silence/music (learned from YouTube-style subtitle data).
# Only dropped when the model is also unsure: a confidently heard "Thank you." in a meeting stays.
_STOCK_PHRASES = {
    "you", "thank you", "thank you very much", "thanks for watching", "thank you for watching",
    "please subscribe", "subtitles by the amara.org community", "bye", "music",
}
_PUNCT = re.compile(r"[^\w\s.]|\.+$")


def is_hallucination(seg: TimedSegment, no_speech_threshold: float, logprob_threshold: float) -> bool:
    # Same rule Whisper uses to discard a window: model thinks it is silence AND is unsure of the text.
    if seg.no_speech_prob > no_speech_threshold and seg.avg_logprob < logprob_threshold:
        return True
    stock = _PUNCT.sub("", normalize_text(seg.text).lower()).strip() in _STOCK_PHRASES
    return stock and math.exp(seg.avg_logprob) < LOW_CONFIDENCE


def clean_segments(
    segments: list[TimedSegment],
    duration: float,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
) -> list[Segment]:
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: s.start):
        text = normalize_text(seg.text)
        if not text or is_hallucination(seg, no_speech_threshold, logprob_threshold):
            continue
        # Whisper repetition loops emit the same line back to back; keep one.
        if out and out[-1].text == text and seg.start - out[-1].end < 1.0:
            out[-1].end = round(max(out[-1].end, min(seg.end, duration)), 3)
            out[-1].words += _words(seg, duration)
            continue
        start = max(0.0, min(seg.start, duration))
        if out:
            start = max(start, out[-1].start)  # keep the timeline monotonic
        end = max(start, min(seg.end, duration))
        confidence = round(min(1.0, math.exp(seg.avg_logprob)), 3)
        out.append(
            Segment(
                id=len(out),
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                confidence=confidence,
                low_confidence=confidence < LOW_CONFIDENCE,
                words=_words(seg, duration),
            )
        )
    return out


def _words(seg: TimedSegment, duration: float) -> list[Word]:
    return [
        Word(
            start=round(min(w.start, duration), 3),
            end=round(min(w.end, duration), 3),
            text=w.text.strip(),
            probability=round(w.probability, 3),
        )
        for w in seg.words
        if w.text.strip()
    ]
