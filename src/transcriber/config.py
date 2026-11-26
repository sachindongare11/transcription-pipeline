"""Runtime configuration. Every knob is overridable via a TRANSCRIBER_* env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PipelineConfig:
    # --- engine ---
    engine: str = "whisper"            # "whisper" | "mock"
    model_size: str = "base"           # tiny/base/small/medium/large-v3 (accuracy vs. latency)
    device: str = "auto"               # cpu/cuda/auto
    compute_type: str = "int8"         # int8 on CPU, float16 on GPU
    beam_size: int = 5
    hotwords: str = ""                 # domain vocabulary biasing, e.g. "DORA ICT CBI"
    language_pin_probability: float = 0.8  # lock the auto-detected language only once this sure

    # --- long-audio handling ---
    chunk_seconds: float = 120.0       # target chunk length fed to the engine
    cut_search_seconds: float = 10.0   # look back this far from the target for a quiet cut point
    silence_rms: float = 0.01          # below this RMS a 30 ms frame counts as silence
    overlap_seconds: float = 1.0       # used only when no silence was found (hard cut)

    # --- content adaptivity (speech vs. music/singing) ---
    vad: str = "auto"                  # on | off | auto: VAD first, retry a chunk without it when
    vad_min_speech_ratio: float = 0.2  #   VAD keeps < 20% of a chunk that is >= 50% non-silent
    min_active_ratio: float = 0.5      #   (i.e. loud but "not speech": music, singing, chanting)
    max_duration_seconds: float = 4 * 3600.0
    chunk_retries: int = 2

    # --- post-processing ---
    no_speech_threshold: float = 0.6   # Whisper's own hallucination heuristic:
    logprob_threshold: float = -1.0    # drop if no_speech_prob > 0.6 AND avg_logprob < -1.0

    # --- service ---
    max_upload_mb: int = 500
    sync_max_seconds: float = 120.0    # longer files are processed as async jobs
    sync_wait_seconds: float = 60.0    # a sync request still queued/running after this gets a 202
    workers: int = 1                   # concurrent transcriptions (model is CPU/GPU bound)
    max_queued_jobs: int = 32
    job_ttl_seconds: float = 3600.0    # finished jobs (and transcripts) are dropped after this

    @classmethod
    def from_env(cls, **overrides) -> "PipelineConfig":
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f"TRANSCRIBER_{f.name.upper()}")
            if raw is not None:
                values[f.name] = type(f.default)(raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
