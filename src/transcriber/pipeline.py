"""Orchestration: probe -> decode/normalise -> chunk -> transcribe -> merge -> clean."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from transcriber.audio import AudioDecodeError, probe, stream_pcm
from transcriber.chunking import Chunk, active_ratio, chunk_stream
from transcriber.config import PipelineConfig
from transcriber.engines import Engine, EngineResult, build_engine
from transcriber.postprocess import TimedSegment, TimedWord, clean_segments
from transcriber.schemas import SourceInfo, Transcript

log = logging.getLogger(__name__)

# (seconds of audio processed so far, total seconds if known)
ProgressFn = Callable[[float, float | None], None]


class AudioTooLongError(ValueError):
    pass


class UnsupportedLanguageError(ValueError):
    pass


class TranscriptionPipeline:
    def __init__(self, config: PipelineConfig | None = None, engine: Engine | None = None):
        self.config = config or PipelineConfig()
        self.engine = engine or build_engine(self.config)

    def check_language(self, language: str | None) -> str | None:
        """Normalise and validate a caller-supplied language code (None = auto-detect)."""
        if language is None or not language.strip():
            return None
        code = language.strip().lower()
        if self.engine.languages is not None and code not in self.engine.languages:
            raise UnsupportedLanguageError(
                f"unsupported language {language!r}; expected an ISO 639-1 code such as 'en'"
            )
        return code

    def run(
        self,
        path: str | Path,
        language: str | None = None,
        filename: str | None = None,
        on_progress: ProgressFn | None = None,
    ) -> Transcript:
        cfg = self.config
        pinned = self.check_language(language)  # fail before any decoding
        self.engine.load()  # one-off model load stays out of processing time / RTF
        started = time.perf_counter()
        info = probe(path)
        if info.duration is not None and info.duration > cfg.max_duration_seconds:
            raise AudioTooLongError(
                f"audio is {info.duration:.0f}s; limit is {cfg.max_duration_seconds:.0f}s"
            )

        chunks = chunk_stream(
            stream_pcm(path),
            chunk_seconds=cfg.chunk_seconds,
            cut_search_seconds=cfg.cut_search_seconds,
            silence_rms=cfg.silence_rms,
            overlap_seconds=cfg.overlap_seconds,
        )

        timed: list[TimedSegment] = []
        duration = 0.0
        n_chunks = 0
        vad_fallbacks = 0
        forced = pinned is not None
        best_language, best_probability = None, 0.0
        for chunk in chunks:
            if chunk.end > cfg.max_duration_seconds:  # header lied or was missing
                raise AudioTooLongError(f"audio exceeds {cfg.max_duration_seconds:.0f}s limit")
            result, fell_back = self._transcribe_chunk(chunk, pinned)
            vad_fallbacks += fell_back
            # Lock the language once detection is confident: per-chunk detection on a quiet
            # or music-only chunk can flip languages mid-transcript. Locking on a *weak*
            # detection is worse -- a music intro could fix the wrong language for hours --
            # so until then each chunk detects for itself and we remember the best guess.
            if pinned is None and result.segments and result.language:
                p = result.language_probability or 0.0
                if p > best_probability:
                    best_language, best_probability = result.language, p
                if p >= cfg.language_pin_probability:
                    pinned = result.language
                    log.info("language pinned to %r (p=%.2f) at chunk %d", pinned, p, chunk.index)
            timed.extend(_to_source_timeline(chunk, result))
            duration = chunk.end
            n_chunks += 1
            log.info("chunk %d [%.1fs-%.1fs] clean_cut=%s vad=%s segments=%d",
                     chunk.index, chunk.start, chunk.end, chunk.clean_cut,
                     "off(fallback)" if fell_back else cfg.vad, len(result.segments))
            if on_progress:
                on_progress(duration, info.duration)

        if n_chunks == 0:
            raise AudioDecodeError("no decodable audio samples")

        segments = clean_segments(
            timed, duration, cfg.no_speech_threshold, cfg.logprob_threshold
        )
        warnings = []
        if vad_fallbacks:
            warnings.append(
                f"{vad_fallbacks}/{n_chunks} chunk(s) had continuous sound but little detected speech "
                "(music or singing?); transcribed without the speech filter, so expect lower accuracy "
                "and possible spurious text"
            )
        language = pinned or best_language
        if not forced and best_language and best_probability < cfg.language_pin_probability:
            warnings.append(
                f"language detection uncertain ({best_language} p={best_probability:.2f}); "
                "pass the language explicitly if known"
            )
        if not segments:
            warnings.append("no speech detected")
        for w in warnings:
            log.warning(w)
        elapsed = time.perf_counter() - started
        return Transcript(
            text=" ".join(s.text for s in segments),
            language=language,
            duration=round(duration, 3),
            segments=segments,
            source=SourceInfo(
                filename=filename or Path(path).name,
                container=info.container,
                codec=info.codec,
                sample_rate=info.sample_rate,
                channels=info.channels,
            ),
            engine=self.engine.name,
            chunks=n_chunks,
            processing_seconds=round(elapsed, 3),
            real_time_factor=round(elapsed / duration, 4) if duration else 0.0,
            warnings=warnings,
        )

    def _transcribe_chunk(self, chunk: Chunk, language: str | None) -> tuple[EngineResult, bool]:
        """Transcribe one chunk; returns (result, whether we fell back to no-VAD)."""
        cfg = self.config
        result = self._call_engine(chunk, language, vad=cfg.vad != "off")
        if cfg.vad != "auto" or result.speech_seconds is None or not chunk.duration:
            return result, False
        # VAD kept almost nothing, yet the chunk is mostly *not* silent: that is music/singing/
        # chanting, not silence. Retrying without VAD beats returning an empty transcript.
        # Genuinely quiet chunks (pauses, room tone) fail the energy test and are not retried,
        # so the hallucination protection VAD gives speech audio is kept.
        if (result.speech_seconds / chunk.duration < cfg.vad_min_speech_ratio
                and active_ratio(chunk.samples, cfg.silence_rms) >= cfg.min_active_ratio):
            log.info("chunk %d: VAD kept %.1fs of %.1fs of non-silent audio; retrying without VAD",
                     chunk.index, result.speech_seconds, chunk.duration)
            return self._call_engine(chunk, language, vad=False), True
        return result, False

    def _call_engine(self, chunk: Chunk, language: str | None, vad: bool) -> EngineResult:
        attempts = self.config.chunk_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self.engine.transcribe(chunk.samples, language, vad)
            except Exception:  # engine errors are opaque (OOM, network for hosted APIs, ...)
                if attempt == attempts:
                    raise
                log.warning("chunk %d failed (attempt %d/%d), retrying",
                            chunk.index, attempt, attempts, exc_info=True)
                time.sleep(0.5 * attempt)
        raise AssertionError("unreachable")


def _to_source_timeline(chunk: Chunk, result: EngineResult) -> list[TimedSegment]:
    """Shift a chunk's output onto the source timeline, keeping only what this chunk owns.

    Ownership is decided per word (by its midpoint): Whisper segments span several
    seconds, so a segment-level rule keeps the overlap words twice -- once at the end
    of one chunk and again at the start of the next. Segment bounds are then rebuilt
    from the surviving words, which also trims pauses Whisper folds into segment ends.
    """
    out = []
    for s in result.segments:
        if s.words:
            words = [
                TimedWord(chunk.start + w.start, chunk.start + w.end, w.text, w.probability)
                for w in s.words
                if chunk.owns(chunk.start + (w.start + w.end) / 2)
            ]
            if not words:
                continue
            start, end, text = words[0].start, words[-1].end, "".join(w.text for w in words)
        else:  # engine without word timing: best effort at segment granularity
            start, end, text, words = chunk.start + s.start, chunk.start + s.end, s.text, []
            if not chunk.owns((start + end) / 2):
                continue
        out.append(TimedSegment(start, end, text, s.avg_logprob, s.no_speech_prob, words))
    return out
