# Transcription pipeline

Turns an audio file into a timestamped transcript ready for downstream use (JSON, SRT,
VTT or plain text). It runs as a **CLI** or an **HTTP service**. Speech-to-text is
open-source Whisper via [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
running locally on CPU. No model is trained here: the work is the engineering around the model.

**Contents:**
[Quick start](#quick-start) ·
[Approach](#1-approach) ·
[Design decisions](#2-key-design-and-technical-decisions) ·
[Code structure](#3-code-structure-and-implementation) ·
[Assumptions and trade-offs](#4-assumptions-considerations-and-trade-offs) ·
[Results](#5-results) ·
[Production path](#6-limitations-and-production-path)

---

## Quick start

Requires Python 3.10+. FFmpeg is **not** needed: PyAV bundles its libraries.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/make_samples.py samples   # synthetic speech in 5 formats + a 5-minute file (macOS `say`)
```

CLI:

```bash
.venv/bin/transcribe samples/sample.mp3 -f txt
.venv/bin/transcribe samples/long.mp3 -f srt -o long.srt -v      # long file: shows chunking progress
.venv/bin/transcribe samples/sample.wav --hotwords "DORA ICT"     # domain vocabulary biasing
```

HTTP service (interactive docs at `http://localhost:8000/docs`):

```bash
.venv/bin/uvicorn --factory transcriber.api:create_app --port 8000
```

```bash
curl -F file=@samples/sample.m4a -F format=srt localhost:8000/v1/transcriptions   # short: 200, transcript inline
curl -F file=@samples/long.mp3 localhost:8000/v1/transcriptions                     # long: 202 + status_url
curl localhost:8000/v1/jobs/<id>                                                    # status, progress, transcript
curl "localhost:8000/v1/jobs/<id>?format=vtt"
```

Tests and evaluation:

```bash
.venv/bin/pytest                       # 42 tests, mock engine, no model download (~2 s)
.venv/bin/pytest -m slow               # real Whisper round trip
.venv/bin/python scripts/eval_wer.py   # word error rate per configuration against a known script
```

The first real transcription downloads the Whisper `base` model (~145 MB). Every setting in
[config.py](src/transcriber/config.py) can be overridden with a `TRANSCRIBER_*` environment
variable, for example `TRANSCRIBER_MODEL_SIZE=small` or `TRANSCRIBER_DEVICE=cuda`.

---

## 1. Approach

**Framing the problem.** Speech recognition itself is a solved component: an open-source
model does it well. What makes a *pipeline* reliable is everything around the model:

- accepting arbitrary, sometimes broken input safely;
- making every audio format look the same to the model;
- keeping memory, latency and the cost of a failure bounded when files are long;
- keeping timestamps correct when audio is split into pieces;
- producing output that downstream systems can trust, or at least know when not to trust.

So I spent the effort there, and measured it: a synthetic test file with a known script
gives an exact word error rate (WER), which turned design arguments into numbers.

**Pipeline, end to end.**

```
 file / upload
      │
      ▼
 ingest          size cap while streaming · validate by content, not extension
      │
      ▼
 decode          any format → 16 kHz mono float32 (PyAV), in small blocks
      │
      ▼
 chunk           ~2 min chunks, cut at the quietest point; overlap only when there is no silence
      │
      ▼
 transcribe      Whisper per chunk · adaptive speech filter · retry · language lock
      │
      ▼
 merge           chunk times → source times · overlap resolved word by word
      │
      ▼
 clean           drop invented text · collapse repeats · confidence · warnings
      │
      ▼
 JSON / SRT / VTT / TXT
```

Everything from decoding to transcription is a chain of streaming generators, so only about
one chunk of audio is in memory at a time, however long the file.

**How I worked.** I built the pipeline end to end first and tested it on real speech
(synthetic voice in five formats, plus a 5-minute file) and on a pop song. Then I reviewed
it against the use case with WER measurements. That review found defects the unit tests had
missed, the most important being duplicated words at chunk joins. All were fixed, each
covered by a test.

---

## 2. Key design and technical decisions

### Speech-to-text engine: faster-whisper behind a small interface
- **Why Whisper:** strong multilingual accuracy, open-source, and it runs **fully offline**,
  so confidential audio never leaves the host.
- **Why faster-whisper:** the same Whisper weights through CTranslate2 with int8 quantisation.
  Its published benchmarks report about 4× the speed of the reference implementation (I did
  not re-measure that). Measured here: `base` runs about 33× faster than real time on CPU.
- **Why an interface:** the pipeline only calls `transcribe(pcm, language, vad)`. A hosted API
  (Azure Speech, Deepgram) or a GPU server is one new class. A `MockEngine` that produces
  Whisper-shaped output lets the tests run in seconds without model weights.

### Convert every format once, at the edge
- Every input is decoded and resampled to **16 kHz mono float32**, the format Whisper was
  trained on. Nothing downstream ever branches on format.
- **PyAV** bundles FFmpeg's libraries: no system binary to install or patch. It covers WAV,
  MP3, AAC/M4A, FLAC, Ogg/Opus, WebM and audio tracks in video files.
- Files are identified **by content, never extension**: FLAC renamed to `.mp3` decodes, and a
  text file renamed to `.mp3` is rejected (`415`).

### Long audio: stream, cut on silence, merge word by word
Whisper only looks at 30 seconds at a time, so long audio is a systems problem, not a model one.
- **Streaming** keeps memory to about one chunk. A 3-hour file would otherwise be ~690 MB in memory.
- **Silence-aware cuts:** near each 2-minute mark, the chunker looks back up to 10 s for the
  quietest 30 ms frame and cuts there, so words are not split.
  Measured: **2.0% WER vs 5.3%** for blind cuts at the same chunk size.
- **Overlap when there is no silence** (continuous speech, music): the next chunk starts 1 s
  earlier for context. Each chunk owns a time window, split at the *middle* of the overlap,
  and ownership is decided **per word**. My first version decided per *segment* and
  duplicated words at every join (11.2% WER); per word gives **2.35%**.
- **Per-chunk retry** means one failure doesn't lose a 2-hour job.

### Adapt the speech filter to the content
- Whisper invents text over silence, so a speech filter (Silero VAD) is on by default.
- The filter is trained on talking. On a pop song it kept 0.8 s of 113 s, and the transcript
  came back **empty with no explanation**.
- Now it adapts per chunk. If the filter keeps under 20% of a chunk that is at least 50% loud
  by energy (music, singing), that chunk is re-run without the filter and a warning is
  recorded. Quiet chunks fail the energy test, so they keep the protection.

### Timestamps rebuilt from words
- Word-level timing is on. Segment start and end are rebuilt from the first and last word,
  because Whisper's own segment ends absorb trailing pauses. One sentence went from 10.4 s to 4.4 s.
- Word timing is also what makes the overlap merge exact. Cost: about 3% extra processing time.

### Output that explains itself
- Each segment has absolute timestamps, text, `confidence`, a `low_confidence` flag and
  `words[]`, so downstream systems can route weak segments to human review.
- Clean-up:
  - Whisper's own no-speech rule;
  - known stock phrases Whisper invents (like "Thank you."), dropped **only when confidence
    is also low**, so a clearly spoken "Thank you." survives;
  - repetition loops collapsed;
  - timestamps kept in order and within the audio length.
- Every transcript carries `warnings`: music fallback, uncertain language, no speech detected.
  An empty or weak result is never a silent failure.

### Language: detect, but lock only when confident
- Locking the detected language avoids flip-flopping between chunks. Locking on a *weak* guess
  is worse: a music intro could fix the wrong language for hours.
- So the language is locked only at probability ≥ 0.8. Until then each chunk detects for
  itself, and the best guess is reported with a warning.
- A caller-supplied code is normalised and checked against Whisper's 100 languages (`400` if invalid).

### API: asynchronous for anything long
- Short clips (≤ 2 min) can be synchronous. A synchronous request waits at most 60 s, then
  gets a job to poll instead.
- Longer audio returns `202 Accepted` plus a job URL to poll for progress (0 → 1) and the
  result in any format, because long requests die at proxy timeouts.
- An explicit synchronous request for long audio is refused (`400`), so nobody can hold a
  connection open for an hour.
- **Upload size is enforced while the body streams in** (ASGI middleware). FastAPI's
  multipart parser otherwise writes the whole file to disk *before* the endpoint runs, so a
  check in the endpoint would be too late.
- Cheapest checks first: size (while streaming) → language → empty → decodable → duration →
  queue capacity, so bad requests fail in milliseconds without taking a worker.
- `/healthz` reports liveness. `/readyz` returns `503` until a background warm-up has loaded
  the model.

---

## 3. Code structure and implementation

```
src/transcriber/
  audio.py        probe() validates by content; stream_pcm() decodes any format to 16 kHz mono, block by block
  chunking.py     silence-aware chunker; overlap on forced cuts; each chunk's ownership window
  engines.py      Engine interface; FasterWhisperEngine (real) and MockEngine (tests)
  pipeline.py     orchestration: validate → chunk → transcribe (retry, VAD fallback, language lock) → merge → clean
  postprocess.py  invented-text filter, repetition collapse, confidence, word timing
  schemas.py      output contract (Pydantic): Transcript, Segment, Word
  formats.py      JSON / SRT / VTT / TXT rendering
  config.py       every setting, overridable via TRANSCRIBER_* env vars
  cli.py          `transcribe` command
  api.py          FastAPI app: endpoints, sync/async handling, readiness
  limits.py       ASGI middleware enforcing the upload size limit while streaming
  jobs.py         job store: TTL eviction, active-job cap (the one class to replace to scale out)
scripts/
  make_samples.py synthetic speech in several formats + a long file
  eval_wer.py     word error rate per configuration against the known script
tests/            42 fast tests (mock engine) + 1 real-model test
Dockerfile        model weights baked into the image; runs as non-root
```

**How a request flows:**
1. `api.py` validates the request, saves the upload, and creates a job.
2. A worker thread calls `TranscriptionPipeline.run()` in `pipeline.py`.
3. `run()` calls `probe()`, then `stream_pcm()`, and feeds the blocks into `chunk_stream()`.
4. For each chunk, `_transcribe_chunk()` calls the engine (with retry and the adaptive filter).
   `_to_source_timeline()` then keeps only the words this chunk owns, shifted to source time.
5. `clean_segments()` produces the final segments, and `formats.py` renders the requested format.

**Output contract** (abbreviated):

```json
{
  "text": "Good morning everyone, and welcome to ...",
  "language": "en",
  "duration": 12.808,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.08, "text": "Good morning everyone, and welcome to the quarterly regulatory review.",
     "confidence": 0.748, "low_confidence": false,
     "words": [{"start": 0.0, "end": 0.18, "text": "Good", "probability": 0.887}, "..."]}
  ],
  "source": {"filename": "sample.mp3", "container": "mp3", "codec": "mp3float", "sample_rate": 44100, "channels": 2},
  "engine": "faster-whisper/base", "chunks": 1, "processing_seconds": 1.427, "real_time_factor": 0.1115,
  "warnings": []
}
```

**Testing strategy:**
- **Mechanics are tested with a mock engine.** It derives timestamps from signal energy, so
  tests can place sounds at known times and assert they come back within 50 ms, through MP3
  encoding and across chunk joins.
- **The mock reproduces the shape of Whisper's output** (multi-word segments with word timing).
  An earlier mock emitted one segment per sound and could not catch the duplicated-word bug;
  the current seam test fails against the old merge logic.
- **Accuracy is measured separately** against a known script (`scripts/eval_wer.py`), because
  unit tests cannot measure it.

---

## 4. Assumptions, considerations and trade-offs

**Assumptions**
- The input is mostly speech; music is handled but transcribed with lower accuracy.
- One audio stream per file. Multi-channel audio is downmixed to mono.
- English was the test language; the model itself supports 100 languages.
- The service runs as a **single process** (see the trade-off on the job store below).
- Accuracy numbers come from **clean, single-speaker synthetic speech**. Real recordings
  (accents, noise, crosstalk, phone audio) will score worse.

**Trade-offs**
- **Self-hosted vs hosted API:**
  - Self-hosting wins on data residency, cost at volume and control of the model.
  - Hosted wins on time to market, scaling, and built-in speaker labels and streaming.
  - The engine interface keeps this reversible.
- **Model size:** `base` is 1.76% WER at 0.030 real-time factor; `tiny` is 5.29% at 0.017.
  That's about half the compute for 3× the errors. `large-v3` on a GPU is the accuracy choice
  for production.
- **Chunk length (2 min):**
  - Longer chunks give more context and fewer joins.
  - Shorter chunks give finer retry, progress and parallelism.
  - Measured: 1.76% WER at 120 s vs 1.96% at 20 s.
- **Speech filter on by default:** protects against invented text but loses sung vocals; the
  adaptive fallback costs one extra (cheap) filtered pass on music chunks.
- **Hotwords:**
  - They fixed misheard acronyms ("Iced" → "ICT").
  - They made segment boundaries coarser.
  - The 0% WER result was measured on the same audio the words were chosen for, so it proves
    the mechanism, not generalisation.
- **Word timestamps:** about 3% extra compute, in exchange for exact chunk joins and tighter timestamps.
- **Downmixing to mono:** right for mixed recordings; wrong when each channel is a different
  speaker (call-centre audio), which should be split per channel.
- **In-process job store and thread pool:**
  - They keep the service to one command.
  - The cost is a single worker process, and jobs are lost on restart.
  - Finished jobs expire after 1 hour, and active jobs are capped.
- **Multipart upload through the API:** simple for clients; at scale, clients should upload
  straight to object storage via pre-signed URLs.

**Constraints and known gaps**
- Confidence is the model's own probability and is not calibrated against real errors.
- The invented-phrase filter is English-only.
- Word timestamps are approximate (roughly ±100–200 ms).
- Not yet built: speaker diarization, number normalisation (the output mixes "four hours" and
  "4 hours"), redaction of personal data, authentication and rate limiting.

---

## 5. Results

Apple M4, CPU only, Whisper `base` int8.

| Input | Result |
|---|---|
| 12.8 s clip as WAV / MP3 / FLAC / M4A / OGG | Same transcript from every format |
| 292 s MP3 with pauses and noise | 3 chunks, both cuts in silence (115.3 s, 232.5 s), all 42 sentences exactly once |
| Text file renamed `.mp3` | Rejected: HTTP 415 / CLI exit 2 |
| 290 s pop song | Speech filter alone: empty transcript. With adaptive fallback: 21 segments from the vocal entry, plus warnings |
| Acronym "ICT" | Heard as "Iced"; correct with `--hotwords "DORA ICT"` |

Word error rate on the 292 s file (510-word known script, `scripts/eval_wer.py`; model load excluded from RTF):

| Configuration | WER | Words out | RTF |
|---|---|---|---|
| base, silence-aware 120 s chunks (default) | 1.76% | 510 | 0.030 |
| base + hotwords "DORA ICT" | 0.00% | 510 | 0.030 |
| tiny | 5.29% | 499 | 0.017 |
| base, silence-aware 20 s chunks | 1.96% | 508 | 0.037 |
| base, blind 20 s cuts, no overlap | 5.29% | 496 | 0.041 |
| base, blind 20 s cuts + 1 s overlap | 2.35% | 507 | 0.043 |

---

## 6. Limitations and production path

What I would change before real traffic:

1. **Uploads:** clients upload directly to object storage via pre-signed URLs. The API only
   receives a reference.
2. **Processing:** a queue feeds GPU workers that autoscale on queue depth and transcribe the
   chunks of one file in parallel. The merge already sorts by absolute time, so the order
   chunks finish in doesn't matter.
3. **Job state:** move to Redis or a database (replacing `jobs.py`), and add webhooks instead
   of polling. The API contract stays the same.
4. **Evaluation:** build a labelled, in-domain test set before choosing the model size, and
   track WER on a sampled audit set, the low-confidence rate and real-time factor as service metrics.
5. **Downstream:** add speaker diarization, sentence re-segmentation from word timings, number
   normalisation, and redaction of personal data before storage.
6. **Security and operations:** authentication and per-tenant rate limits at an API gateway,
   tracing per job, and model weights baked into the image (already done in the `Dockerfile`).
