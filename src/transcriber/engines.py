"""Speech-to-text engines behind one small interface.

The pipeline only depends on `Engine.transcribe(pcm, language) -> EngineResult`,
so swapping faster-whisper for a hosted API (Azure Speech, Deepgram, OpenAI)
or a GPU server is a new class here, not a pipeline change. `MockEngine` lets
tests and CI exercise chunking/offset/merge logic without model weights.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from transcriber.audio import SAMPLE_RATE
from transcriber.config import PipelineConfig


@dataclass
class RawWord:
    start: float            # seconds, relative to the chunk passed in
    end: float
    text: str               # Whisper convention: carries its own leading space
    probability: float = 1.0


@dataclass
class RawSegment:
    start: float            # seconds, relative to the chunk passed in
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: list[RawWord] = field(default_factory=list)  # empty if the engine has no word timing


@dataclass
class EngineResult:
    segments: list[RawSegment]
    language: str | None = None
    language_probability: float | None = None
    speech_seconds: float | None = None  # audio the engine judged to be speech (after VAD)


class Engine(Protocol):
    name: str
    languages: frozenset[str] | None  # codes the engine accepts; None = no restriction

    def load(self) -> object: ...

    def transcribe(self, pcm: np.ndarray, language: str | None = None, vad: bool = True) -> EngineResult: ...


@dataclass
class FasterWhisperEngine:
    """Whisper via CTranslate2: ~4x faster than openai-whisper, int8 on CPU, same accuracy."""

    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 5
    hotwords: str = ""
    _model: object = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"faster-whisper/{self.model_size}"

    @property
    def languages(self) -> frozenset[str]:
        from faster_whisper.tokenizer import _LANGUAGE_CODES  # the 100 codes Whisper was trained on

        return frozenset(_LANGUAGE_CODES)

    def load(self):
        # Lazy + locked: the service starts fast, and concurrent first requests load once.
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
        return self._model

    def transcribe(self, pcm: np.ndarray, language: str | None = None, vad: bool = True) -> EngineResult:
        model = self.load()
        segments, info = model.transcribe(
            pcm,
            language=language,
            beam_size=self.beam_size,
            # Bias decoding toward domain terms (acronyms, product names) the base model mishears.
            hotwords=self.hotwords or None,
            # Silero VAD skips non-speech, the main source of hallucinated text. It is trained on
            # talking, so it also discards sung vocals -- the pipeline decides when to turn it off.
            vad_filter=vad,
            # Conditioning on previous text helps fluency but is what lets Whisper fall into
            # repetition loops on long audio; chunk-level context is enough.
            condition_on_previous_text=False,
            # Word timing (cross-attention alignment, ~10-20% extra decode time) lets the
            # pipeline merge chunk seams word by word and tighten segment end times.
            word_timestamps=True,
        )
        # `segments` is a lazy generator -- decoding happens while we iterate.
        raw = [
            RawSegment(
                s.start, s.end, s.text, s.avg_logprob, s.no_speech_prob,
                [RawWord(w.start, w.end, w.word, w.probability) for w in (s.words or [])],
            )
            for s in segments
        ]
        speech = info.duration_after_vad if vad else info.duration
        return EngineResult(raw, info.language, info.language_probability, speech)


class MockEngine:
    """Deterministic fake that behaves like Whisper's output *shape*.

    Every contiguous non-silent run is a "word"; words separated by gaps of at most
    `min_gap` are grouped into one multi-second "segment". Timestamps are real
    (derived from signal energy), and a word clipped by a chunk edge comes back
    shortened -- exactly the situation seam merging has to get right.
    """

    name = "mock"
    languages = None

    def __init__(self, threshold: float = 0.02, min_gap: float = 0.3):
        self.threshold = threshold
        self.min_gap = min_gap
        self.calls = 0

    def load(self) -> None:
        pass

    def transcribe(self, pcm: np.ndarray, language: str | None = None, vad: bool = True) -> EngineResult:
        self.calls += 1
        frame = int(0.02 * SAMPLE_RATE)
        n = len(pcm) // frame
        if n == 0:
            return EngineResult([], language or "en", 1.0, 0.0)
        rms = np.sqrt(np.mean(pcm[: n * frame].reshape(n, frame) ** 2, axis=1))
        words: list[RawWord] = []
        for i in np.flatnonzero(rms > self.threshold):
            t0, t1 = i * frame / SAMPLE_RATE, (i + 1) * frame / SAMPLE_RATE
            if words and t0 - words[-1].end < 1e-6:
                words[-1].end = t1
            else:
                words.append(RawWord(t0, t1, ""))
        segments: list[RawSegment] = []
        for w in words:
            w.text = f" w{w.start:.2f}"
            if segments and w.start - segments[-1].end <= self.min_gap:
                segments[-1].words.append(w)
                segments[-1].end = w.end
            else:
                segments.append(RawSegment(w.start, w.end, "", avg_logprob=-0.1, words=[w]))
        for seg in segments:
            seg.text = "".join(w.text for w in seg.words)
        return EngineResult(segments, language or "en", 1.0, sum(w.end - w.start for w in words))


def build_engine(config: PipelineConfig) -> Engine:
    if config.engine == "mock":
        return MockEngine()
    if config.engine == "whisper":
        return FasterWhisperEngine(
            config.model_size, config.device, config.compute_type, config.beam_size, config.hotwords
        )
    raise ValueError(f"unknown engine {config.engine!r}")
