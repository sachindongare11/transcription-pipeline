import numpy as np
import pytest

from transcriber.audio import SAMPLE_RATE, AudioDecodeError, load_pcm, probe
from conftest import bursts

FORMATS = [
    # name, container, codec, source rate, channels
    ("a.wav", "wav", "pcm_s16le", 44100, 2),
    ("a.mp3", "mp3", "libmp3lame", 44100, 2),
    ("a.flac", "flac", "flac", 48000, 1),
    ("a.m4a", "ipod", "aac", 22050, 2),
    ("a.ogg", "ogg", "libopus", 48000, 1),
    ("a.webm", "webm", "libopus", 48000, 2),
]


@pytest.mark.parametrize("name,fmt,codec,rate,channels", FORMATS)
def test_every_format_normalises_to_16k_mono_float(make_audio, name, fmt, codec, rate, channels):
    src = bursts([(0.5, 1.5)], total=3.0)
    path = make_audio(src, name, fmt, codec, rate=rate, channels=channels)

    info = probe(path)
    assert info.sample_rate == rate and info.channels == channels

    pcm = load_pcm(path)
    assert pcm.dtype == np.float32 and pcm.ndim == 1
    # lossy codecs add encoder priming/padding (AAC: up to ~1-2 frames); within 100 ms
    assert abs(len(pcm) / SAMPLE_RATE - 3.0) < 0.1
    # the burst is still where we put it (energy check, codec-agnostic)
    def rms(a, b):
        return float(np.sqrt(np.mean(pcm[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)] ** 2)))
    assert rms(0.7, 1.3) > 0.1
    assert rms(2.0, 2.8) < 0.01


def test_extension_is_not_trusted(tmp_path, make_audio):
    # real FLAC content with a misleading name still decodes
    path = make_audio(bursts([(0, 1)], 1.0), "really_flac.mp3", "flac", "flac")
    assert probe(path).codec == "flac"
    # and a text file named .mp3 is rejected cleanly
    bogus = tmp_path / "bogus.mp3"
    bogus.write_text("not audio")
    with pytest.raises(AudioDecodeError):
        probe(bogus)
