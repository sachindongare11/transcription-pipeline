"""Word error rate of the pipeline on samples/long.mp3 against its known script.

The sample is synthesised from SCRIPT in make_samples.py, so the reference text
is exact. Each row isolates one engineering choice (model size, vocabulary
biasing, chunking strategy). "blind" rows force cuts at arbitrary points to
stress seam handling, the part of long-audio processing most likely to regress.

    python scripts/make_samples.py samples && python scripts/eval_wer.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_samples import SCRIPT  # noqa: E402

from transcriber import PipelineConfig, TranscriptionPipeline  # noqa: E402

NUM = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "72": "seventy two"}
BLIND = dict(chunk_seconds=20, cut_search_seconds=0.05, silence_rms=-1)  # never finds a clean cut

RUNS = {
    "base, silence-aware 120s chunks (default)": {},
    "base + hotwords 'DORA ICT'": dict(hotwords="DORA ICT"),
    "tiny model": dict(model_size="tiny"),
    "base, silence-aware 20s chunks": dict(chunk_seconds=20, cut_search_seconds=5),
    "base, blind 20s cuts, no overlap": dict(BLIND, overlap_seconds=0),
    "base, blind 20s cuts + 1s overlap": dict(BLIND, overlap_seconds=1),
}


def norm(text: str) -> list[str]:
    text = re.sub(r"[-,.!?;:]", " ", text.lower())
    return " ".join(NUM.get(w, w) for w in text.split()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / len(ref)


def main(audio: str = "samples/long.mp3") -> None:
    ref = norm(" ".join(f"Part {r + 1}. {line}" for r in range(6) for line in SCRIPT))
    print(f"reference: {len(ref)} words\n")
    print(f"{'config':44s} {'WER':>7s} {'words':>6s} {'chunks':>6s} {'RTF':>6s}")
    for name, overrides in RUNS.items():
        pipeline = TranscriptionPipeline(PipelineConfig(**overrides))
        pipeline.engine.load()  # keep model load/download out of the RTF figure
        t = pipeline.run(audio, language="en")
        hyp = norm(t.text)
        print(f"{name:44s} {wer(ref, hyp):7.2%} {len(hyp):6d} {t.chunks:6d} {t.real_time_factor:6.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
