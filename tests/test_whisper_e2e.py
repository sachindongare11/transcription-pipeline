"""Real-model test. Run with: pytest -m slow  (downloads ~145 MB of weights once)."""

import shutil
import subprocess

import pytest

from transcriber.config import PipelineConfig
from transcriber.pipeline import TranscriptionPipeline

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("say") is None, reason="needs macOS `say` to synthesise speech")
def test_real_speech_round_trip(tmp_path):
    wav = tmp_path / "speech.wav"
    subprocess.run(
        ["say", "-o", str(wav), "--data-format=LEI16@22050",
         "The initial notification is due within four hours."],
        check=True,
    )
    t = TranscriptionPipeline(PipelineConfig(model_size="base")).run(wav)

    assert t.language == "en"
    words = t.text.lower()
    assert "notification" in words
    assert "four hours" in words or "4 hours" in words  # Whisper applies its own number formatting
    assert t.segments[0].start < 1.0 and t.segments[-1].end <= t.duration
