import numpy as np

from transcriber.audio import SAMPLE_RATE
from transcriber.chunking import chunk_stream
from conftest import bursts


def blocks(pcm, size=4096):
    return (pcm[i:i + size] for i in range(0, len(pcm), size))


def test_cuts_land_in_silence_and_reassemble_losslessly():
    # 1.8 s "words" every 2 s: one 0.2 s gap per 2 s, so a blind 10 s cut would split a word.
    spans = [(t + 0.1, t + 1.9) for t in np.arange(0, 60, 2.0)]
    pcm = bursts(spans, 60.0)
    chunks = list(chunk_stream(blocks(pcm), chunk_seconds=10, cut_search_seconds=3))

    assert len(chunks) >= 6
    for c in chunks[:-1]:
        assert c.clean_cut
        assert c.duration <= 10.0
        cut = c.end % 2.0  # position inside the 2 s period: must be in the silent gap
        assert cut < 0.1 or cut > 1.9, f"cut at {c.end:.3f}s splits a word"
    # clean cuts => no overlap, contiguous timeline, nothing lost
    for a, b in zip(chunks, chunks[1:]):
        assert np.isclose(a.end, b.start)
        assert np.isclose(a.keep_until, a.end) and np.isclose(b.keep_from, b.start)
    assert np.array_equal(np.concatenate([c.samples for c in chunks]), pcm)


def test_continuous_signal_falls_back_to_hard_cut_with_overlap():
    pcm = bursts([(0, 30)], 30.0)  # no silence anywhere
    chunks = list(chunk_stream(blocks(pcm), chunk_seconds=10, cut_search_seconds=2, overlap_seconds=1))

    assert not chunks[0].clean_cut
    for a, b in zip(chunks, chunks[1:]):
        assert np.isclose(b.start, a.end - 1.0)    # next chunk re-reads 1 s of left context
        # ownership splits the overlap down the middle, with no gap and no double claim
        assert np.isclose(a.keep_until, a.end - 0.5) and np.isclose(b.keep_from, a.keep_until)
    assert np.isclose(chunks[-1].end, 30.0)


def test_short_audio_is_a_single_chunk():
    pcm = bursts([(0.2, 0.8)], 1.0)
    chunks = list(chunk_stream(blocks(pcm), chunk_seconds=120))
    assert len(chunks) == 1 and np.isclose(chunks[0].duration, 1.0)
    assert len(chunks[0].samples) == SAMPLE_RATE
