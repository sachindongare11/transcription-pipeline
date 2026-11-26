import time

import numpy as np
import pytest

from transcriber.config import PipelineConfig
from transcriber.engines import EngineResult, MockEngine
from transcriber.formats import render
from transcriber.pipeline import AudioTooLongError, TranscriptionPipeline
from transcriber.postprocess import TimedSegment, clean_segments
from transcriber.schemas import Segment, SourceInfo, Transcript
from conftest import bursts

SPANS = [(t + 0.5, t + 2.0) for t in np.arange(0, 45, 3.0)]  # 15 "utterances" over 45 s


def pipeline(**cfg):
    return TranscriptionPipeline(PipelineConfig(engine="mock", **cfg), MockEngine())


def assert_matches_spans(segments, spans, tol=0.05):
    assert len(segments) == len(spans)
    for seg, (start, end) in zip(segments, spans):
        assert abs(seg.start - start) < tol and abs(seg.end - end) < tol


def test_timestamps_survive_chunking_and_lossy_encoding(make_audio):
    path = make_audio(bursts(SPANS, 45.0), "long.mp3", "mp3", "libmp3lame", rate=44100, channels=2)
    t = pipeline(chunk_seconds=10, cut_search_seconds=3).run(path)

    assert t.chunks >= 4  # actually exercised the multi-chunk path
    assert_matches_spans(t.segments, SPANS)  # absolute times, not chunk-relative
    assert [s.id for s in t.segments] == list(range(len(SPANS)))
    assert t.source.codec.startswith("mp3") and abs(t.duration - 45.0) < 0.1


def test_hard_cut_overlap_is_not_transcribed_twice(make_audio):
    # one long 25 s "utterance" forces hard cuts; then normal speech
    spans = [(0.0, 25.0), (26.0, 28.0)]
    path = make_audio(bursts(spans, 30.0))
    t = pipeline(chunk_seconds=10, cut_search_seconds=2, overlap_seconds=1).run(path)

    # Mock emits one segment per voiced region per chunk; overlapping copies must be dropped.
    starts = [s.start for s in t.segments]
    assert starts == sorted(starts)
    assert all(b.start >= a.start for a, b in zip(t.segments, t.segments[1:]))
    assert sum(s.end - s.start for s in t.segments) < 30.0 - 1.0 + 0.1  # no double-counted seconds
    assert abs(t.segments[-1].start - 26.0) < 0.05


def test_hard_cut_seams_keep_every_word_exactly_once(make_audio):
    # Continuous "speech": 0.3 s words, 0.1 s gaps -> one long multi-word segment per chunk,
    # and nowhere quiet enough to cut cleanly. Cuts land at arbitrary points, often mid-word.
    words = [(round(t, 2), round(t + 0.3, 2)) for t in np.arange(0.0, 40.0, 0.4)]
    path = make_audio(bursts(words, 40.0))
    cfg = PipelineConfig(chunk_seconds=7, cut_search_seconds=0.05, silence_rms=-1, overlap_seconds=1)
    t = TranscriptionPipeline(cfg, MockEngine()).run(path)

    assert t.chunks >= 5
    out = [(w.start, w.end) for s in t.segments for w in s.words]
    assert len(out) == len(words), "a seam dropped or duplicated words"
    for (s, e), (rs, re_) in zip(out, words):
        assert abs(s - rs) < 0.03 and abs(e - re_) < 0.03, "a clipped edge copy won over the full word"


def test_language_detected_once_then_pinned(make_audio):
    seen = []

    class Recording(MockEngine):
        def transcribe(self, pcm, language=None, vad=True):
            seen.append(language)
            r = super().transcribe(pcm, language, vad)
            r.language = language or "de"
            return r

    path = make_audio(bursts(SPANS, 45.0))
    t = TranscriptionPipeline(PipelineConfig(chunk_seconds=10), Recording()).run(path)
    assert seen[0] is None and set(seen[1:]) == {"de"} and t.language == "de"


class ScriptedLanguage(MockEngine):
    """Reports a scripted (language, probability) per call; records the language it was given."""

    def __init__(self, script):
        super().__init__()
        self.script, self.given = list(script), []

    def transcribe(self, pcm, language=None, vad=True):
        self.given.append(language)
        r = super().transcribe(pcm, language, vad)
        if language is None:
            r.language, r.language_probability = self.script.pop(0)
        return r


def test_low_confidence_detection_is_not_pinned(make_audio):
    # a music intro detects "de" weakly; real speech later detects "en" confidently
    engine = ScriptedLanguage([("de", 0.5), ("en", 0.95)])
    t = TranscriptionPipeline(PipelineConfig(chunk_seconds=10), engine).run(make_audio(bursts(SPANS, 45.0)))
    assert engine.given[:3] == [None, None, "en"]  # kept detecting until sure, then locked
    assert t.language == "en" and not any("uncertain" in w for w in t.warnings)


def test_never_confident_language_is_reported_with_a_warning(make_audio):
    engine = ScriptedLanguage([("de", 0.4), ("nl", 0.6)] + [("de", 0.5)] * 10)
    t = TranscriptionPipeline(PipelineConfig(chunk_seconds=10), engine).run(make_audio(bursts(SPANS, 45.0)))
    assert None in engine.given[1:]       # never locked
    assert t.language == "nl"             # best guess seen
    assert any("uncertain (nl p=0.60)" in w for w in t.warnings)


def test_model_load_is_excluded_from_processing_time(make_audio):
    class SlowLoad(MockEngine):
        loaded = False

        def load(self):
            if not self.loaded:
                time.sleep(0.5)
                self.loaded = True

    t = TranscriptionPipeline(PipelineConfig(), SlowLoad()).run(make_audio(bursts([(1, 2)], 3.0)))
    assert t.processing_seconds < 0.4


def test_transient_engine_failure_is_retried(make_audio):
    class Flaky(MockEngine):
        def transcribe(self, pcm, language=None, vad=True):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient")
            return super().transcribe(pcm, language, vad)

    engine = Flaky()
    path = make_audio(bursts(SPANS, 45.0))
    t = TranscriptionPipeline(PipelineConfig(chunk_seconds=10, chunk_retries=1), engine).run(path)
    assert len(t.segments) == len(SPANS)


class SpeechOnlyVad(MockEngine):
    """Simulates Silero VAD on music: with VAD on it hears 'no speech' in anything."""

    def __init__(self):
        super().__init__()
        self.vad_calls = []

    def transcribe(self, pcm, language=None, vad=True):
        self.vad_calls.append(vad)
        if vad:
            return EngineResult([], "en", 0.5, speech_seconds=0.0)
        return super().transcribe(pcm, language, vad)


def test_music_like_chunk_falls_back_to_no_vad(make_audio):
    path = make_audio(bursts([(0, 30)], 30.0))  # 30 s of continuous sound, VAD rejects all of it
    engine = SpeechOnlyVad()
    t = TranscriptionPipeline(PipelineConfig(chunk_seconds=10), engine).run(path)

    assert engine.vad_calls == [True, False] * t.chunks  # each chunk: try VAD, then retry without
    assert t.segments and t.segments[-1].end > 29.0      # got output across the whole file
    assert any("music or singing" in w for w in t.warnings)


def test_quiet_chunk_does_not_fall_back(make_audio):
    # mostly silence with one short blip: VAD finding nothing is *correct*, no retry
    path = make_audio(bursts([(1.0, 1.5)], 20.0))
    engine = SpeechOnlyVad()
    t = TranscriptionPipeline(PipelineConfig(chunk_seconds=10), engine).run(path)
    assert engine.vad_calls == [True] * t.chunks
    assert t.segments == [] and "no speech detected" in t.warnings


def test_vad_off_and_on_modes_are_respected(make_audio):
    path = make_audio(bursts([(0, 30)], 30.0))
    for mode, expected in (("off", False), ("on", True)):
        engine = SpeechOnlyVad()
        TranscriptionPipeline(PipelineConfig(chunk_seconds=10, vad=mode), engine).run(path)
        assert set(engine.vad_calls) == {expected}


def test_rejects_audio_over_limit(make_audio):
    path = make_audio(bursts([], 5.0))
    with pytest.raises(AudioTooLongError):
        pipeline(max_duration_seconds=2).run(path)


def test_postprocess_filters_hallucinations_and_repetition():
    segs = [
        TimedSegment(0.0, 1.0, "  hello   world ", -0.1, 0.0),
        TimedSegment(1.0, 2.0, "Thanks for watching!", -1.5, 0.9),  # classic silence hallucination
        TimedSegment(2.0, 3.0, "again", -0.2, 0.1),
        TimedSegment(3.1, 4.0, "again", -0.2, 0.1),                   # repetition loop
        TimedSegment(4.0, 5.0, "   ", -0.1, 0.0),
        TimedSegment(5.0, 9.0, "mumble", -2.0, 0.1),                   # kept but flagged
        TimedSegment(9.0, 9.5, "Thank you very much.", -1.4, 0.2),    # stock phrase, unsure -> drop
        TimedSegment(9.5, 9.9, "You", -0.9, 0.3),                       # stock phrase, unsure -> drop
        TimedSegment(9.9, 10.0, "Thank you.", -0.05, 0.0),             # said clearly -> keep
    ]
    out = clean_segments(segs, duration=10.0)
    assert [s.text for s in out] == ["hello world", "again", "mumble", "Thank you."]
    assert out[1].end == 4.0
    assert out[2].end == 9.0 and out[2].low_confidence


def test_subtitle_rendering():
    seg = Segment(id=0, start=3661.25, end=3662.5, text="hi", confidence=0.9)  # > 1 h: hour field
    src = SourceInfo(container="wav", codec="pcm_s16le", sample_rate=16000, channels=1)
    t = Transcript(text="hi", language="en", duration=3663.0, segments=[seg], source=src,
                   engine="mock", chunks=1, processing_seconds=1.0, real_time_factor=0.0)
    srt = render(t, "srt")
    assert srt.startswith("1\n01:01:01,250 --> 01:01:02,500\n")
    assert render(t, "vtt").startswith("WEBVTT\n\n01:01:01.250 --> 01:01:02.500\n")
