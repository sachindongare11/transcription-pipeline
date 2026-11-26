# Transcription pipeline

Audio file in → timestamped, cleaned transcript out (JSON / SRT / VTT / TXT),
as a **CLI** and an **HTTP service**. Speech-to-text is open-source Whisper
running through [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2, int8 on CPU). No model training. The work is all in the
engineering around the model.

```
            ┌──────────┐   ┌───────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
 upload ──▶ │  ingest  │──▶│ decode +      │──▶│ silence-aware│──▶│ STT engine    │──▶│ merge + clean│──▶ JSON/SRT/VTT
 (any fmt)  │ size cap │   │ normalise     │   │ chunker      │   │ (per chunk,   │   │ offsets,     │
            │ probe by │   │ 16 kHz mono   │   │ bounded mem, │   │  retry,       │   │ dedupe seams,│
            │ content  │   │ float32 (PyAV)│   │ overlap on   │   │  lang pinned) │   │ hallucination│
            └──────────┘   └───────────────┘   │ hard cuts    │   └───────────────┘   │ filter, conf.│
                                    streaming generator ─────────────────────────▶  └──────────────┘
```

| Module | Responsibility |
|---|---|
| [audio.py](src/transcriber/audio.py) | `probe()` validates by content; `stream_pcm()` decodes any format to 16 kHz mono float32, block by block |
| [chunking.py](src/transcriber/chunking.py) | Splits the stream into ~N-second chunks, cutting at the quietest point; overlaps only when forced to hard-cut, and gives each chunk an ownership window |
| [engines.py](src/transcriber/engines.py) | `Engine` protocol; `FasterWhisperEngine` (real) and `MockEngine` (deterministic, for tests/CI) |
| [pipeline.py](src/transcriber/pipeline.py) | Orchestration: probe → chunk → transcribe with retry → map to source timeline, merging seams **word by word** → clean |
| [postprocess.py](src/transcriber/postprocess.py) | Whitespace normalisation, hallucination filter, repetition-loop collapse, monotonic timestamps, confidence |
| [api.py](src/transcriber/api.py) | FastAPI: sync for short audio, 202 + job polling for long audio, backpressure |
| [limits.py](src/transcriber/limits.py) | ASGI middleware: request-body cap enforced **while the body streams in** |
| [jobs.py](src/transcriber/jobs.py) | Job store: TTL eviction, active-job cap; the one class to swap for Redis/DB to scale out |
| [cli.py](src/transcriber/cli.py) | `transcribe file.mp3 -f srt` |

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/make_samples.py samples      # mock speech via macOS `say`, in 5 formats + a 5-min file
```

CLI:

```bash
.venv/bin/transcribe samples/sample.mp3 -f txt
.venv/bin/transcribe samples/long.mp3 -f srt -o long.srt -v
.venv/bin/transcribe samples/sample.wav --hotwords "DORA ICT"   # domain vocabulary biasing
```

Service:

```bash
.venv/bin/uvicorn --factory transcriber.api:create_app --port 8000
```

```bash
curl -F file=@samples/sample.m4a -F format=srt localhost:8000/v1/transcriptions   # 200, transcript inline
curl -F file=@samples/long.mp3 localhost:8000/v1/transcriptions                     # 202 + status_url
curl localhost:8000/v1/jobs/<id>                                                    # status, progress, transcript
curl localhost:8000/readyz                                                          # 503 until the model is loaded
curl "localhost:8000/v1/jobs/<id>?format=vtt"
```

Tests:

```bash
.venv/bin/pytest            # 42 fast tests, mock engine, no model download (<1 s)
.venv/bin/pytest -m slow    # real Whisper round trip on synthesised speech
.venv/bin/python scripts/eval_wer.py   # WER per config against the known script (regression check)
```

Every setting in [config.py](src/transcriber/config.py) can be overridden with a `TRANSCRIBER_*` env var
(e.g. `TRANSCRIBER_MODEL_SIZE=small`, `TRANSCRIBER_DEVICE=cuda TRANSCRIBER_COMPUTE_TYPE=float16`).

### Output contract

```json
{
  "text": "Good morning everyone, and welcome to ...",
  "language": "en",
  "duration": 12.808,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.08, "text": "Good morning everyone, and welcome to the quarterly regulatory review.",
     "confidence": 0.748, "low_confidence": false,
     "words": [{"start": 0.0, "end": 0.18, "text": "Good", "probability": 0.887},
               {"start": 0.18, "end": 0.5, "text": "morning", "probability": 0.827}, "..."]}
  ],
  "source": {"filename": "sample.mp3", "container": "mp3", "codec": "mp3float", "sample_rate": 44100, "channels": 2},
  "engine": "faster-whisper/base", "chunks": 1, "processing_seconds": 1.427, "real_time_factor": 0.1115,
  "warnings": []
}
```

Timestamps are always **absolute seconds in the source file**, whatever the chunking.
`confidence` (= exp(avg log-prob)) and `low_confidence` let downstream consumers send
weak segments to human review instead of trusting them blindly.

## Measured (Apple M4, CPU, `base` int8)

| Input | Result |
|---|---|
| 12.8 s clip as WAV / MP3 / FLAC / M4A(AAC) / OGG(Opus) | All transcribed with the same text, language `en`, RTF ≈ 0.11 (≈ 9× real time) |
| 292 s MP3 (6 × 7 sentences, pauses, 4 s noise stretches) | 3 chunks, **both cuts landed in silence** (115.3 s, 232.5 s), all 42 sentences present once, RTF 0.03 |
| Text file renamed `.mp3` | Rejected: HTTP 415 / CLI exit 2 |
| 290 s pop song (soft vocals over instruments) | VAD alone kept 0.8 s → **empty transcript**. With adaptive fallback: 21 lyric segments, first word at 33.7 s (vocal entry), a warning in the output, RTF 0.04. Lyric accuracy is mediocre with `base`, a model limit |
| "ICT" (acronym) | Base model hears "Iced"; with `--hotwords "DORA ICT"` → "DORA", "ICT" correct |

**Word error rate** on the 292 s file (510-word known script, `scripts/eval_wer.py`; RTF excludes model load):

| Config | WER | Words out | RTF |
|---|---|---|---|
| base, silence-aware 120 s chunks (default) | 1.76 % | 510 | 0.030 |
| base + hotwords "DORA ICT" | **0.00 %** | 510 | 0.030 |
| tiny | 5.29 % | 499 | 0.017 |
| base, silence-aware 20 s chunks | 1.96 % | 508 | 0.037 |
| base, blind 20 s cuts, no overlap | 5.29 % | 496 | 0.041 |
| base, blind 20 s cuts + 1 s overlap | 2.35 % | 507 | 0.043 |

Reading it: cutting on silence beats blind cutting at the same chunk size (1.96 % vs 5.29 %).
When a blind cut is unavoidable, the overlap recovers most of the loss (2.35 %). Domain
vocabulary is the cheapest accuracy win. The `tiny` model halves compute for roughly 3× the errors.

## Production path (what I'd change before real traffic)

The in-process job store ([jobs.py](src/transcriber/jobs.py)) and thread pool keep this
runnable with one command. They are bounded: finished jobs are evicted after `job_ttl_seconds`,
and active jobs are capped. They are also deliberately **single-process**: run one uvicorn
worker, because with several, a status poll can land on a process that doesn't know the job.
At scale:

1. **Upload → object storage** (pre-signed URL; the API never proxies big bodies) → **queue message** →
   **autoscaled GPU workers** (KEDA on queue depth) → results in a DB, with a **webhook** in place of polling.
   The job model and `/v1/jobs/{id}` contract stay the same.
2. **Fan out chunks** across workers for long files. The chunker already gives independent, offset-tagged units,
   and the merge is order-independent (sorted by absolute time).
3. **Evaluate before tuning**: a WER/CER harness on a labelled in-domain set, per model size,
   with and without hotwords. That's what decides `base` vs `large-v3`, not intuition. Track
   `low_confidence` rate and RTF as production metrics.
4. **Downstream processing**: speaker diarization (pyannote) for meetings; inverse text
   normalisation for consistent numbers (the model already wrote "four hours" as "4 hours");
   PII redaction before storage; word timestamps for search and highlight.
5. **Ops**: bake model weights into the image (done in the [Dockerfile](Dockerfile)), so no
   cold-start download or egress is needed; structured logs per chunk (already logged:
   offsets, clean/hard cut, segment count); OpenTelemetry traces per job.
