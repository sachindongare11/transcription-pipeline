"""HTTP service.

POST /v1/transcriptions   multipart upload; short audio -> 200 with transcript,
                          long audio (or mode=async) -> 202 with a job id
GET  /v1/jobs/{id}        status/progress; transcript once done
GET  /healthz             liveness: the process is up
GET  /readyz              readiness: model loaded, safe to route traffic (503 until then)

Run: uvicorn --factory transcriber.api:create_app

Long files are never processed on the request thread: an HTTP request that
runs for 20 minutes dies at the first proxy/load-balancer timeout. Jobs run on
a bounded worker pool (the model is compute bound, so more threads than
cores/GPUs only adds contention) with a bounded queue for backpressure.

Upload size is capped while the body streams in (limits.BodySizeLimit), not
after the framework has spooled it to disk. Job state lives in jobs.py: in
process, TTL-evicted, single worker process by design -- see that module for
the scale-out path (object storage -> queue -> GPU workers -> Redis/DB).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from transcriber.audio import AudioDecodeError, probe
from transcriber.config import PipelineConfig
from transcriber.engines import Engine
from transcriber.formats import MEDIA_TYPES, render
from transcriber.jobs import InMemoryJobStore, Job, QueueFullError, Status
from transcriber.limits import BodySizeLimit
from transcriber.pipeline import AudioTooLongError, TranscriptionPipeline, UnsupportedLanguageError
from transcriber.schemas import Transcript

log = logging.getLogger(__name__)
Format = Literal["json", "txt", "srt", "vtt"]


def create_app(config: PipelineConfig | None = None, engine: Engine | None = None) -> FastAPI:
    config = config or PipelineConfig.from_env()
    pipeline = TranscriptionPipeline(config, engine)
    executor = ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="transcribe")
    jobs = InMemoryJobStore(config.job_ttl_seconds, config.max_queued_jobs)
    max_bytes = config.max_upload_mb * 1024 * 1024

    model_state = {"ready": False, "error": None}

    def warm_up() -> None:
        # Load (and, without a baked image, download) the model in the background so the
        # server answers liveness probes immediately but receives no traffic until ready.
        try:
            pipeline.engine.load()
            model_state["ready"] = True
            log.info("model %s loaded", pipeline.engine.name)
        except Exception as exc:
            log.exception("model load failed")
            model_state["error"] = f"{type(exc).__name__}"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        threading.Thread(target=warm_up, name="model-warmup", daemon=True).start()
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Transcription service", version="0.1.0", lifespan=lifespan)
    # Multipart overhead is small next to audio; allow 1 MB of slack over the file cap.
    app.add_middleware(BodySizeLimit, max_bytes=max_bytes + (1 << 20),
                       detail=f"upload exceeds {config.max_upload_mb} MB")

    def run_job(job: Job, path: Path) -> None:
        job.status = Status.running

        def progress(done: float, total: float | None) -> None:
            job.progress = min(done / total, 1.0) if total else 0.0

        try:
            job.transcript = pipeline.run(path, job.language, job.filename, progress)
            job.progress = 1.0
            job.finish(Status.succeeded)
        except (AudioDecodeError, AudioTooLongError) as exc:
            job.finish(Status.failed, str(exc))
        except Exception as exc:  # surface, never swallow silently
            log.exception("job %s failed", job.id)
            job.finish(Status.failed, f"internal error: {type(exc).__name__}")
        finally:
            path.unlink(missing_ok=True)

    def respond(transcript: Transcript, fmt: str) -> Response:
        if fmt == "json":
            return JSONResponse(transcript.model_dump())
        return Response(render(transcript, fmt), media_type=MEDIA_TYPES[fmt])

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        body = {"ready": model_state["ready"], "engine": pipeline.engine.name, "error": model_state["error"]}
        return JSONResponse(body, status_code=200 if model_state["ready"] else 503)

    @app.post("/v1/transcriptions")
    def create_transcription(  # sync def: FastAPI runs it in its threadpool, so blocking I/O is fine
        file: UploadFile = File(...),
        language: str | None = Form(None),
        format: Format = Form("json"),
        mode: Literal["auto", "sync", "async"] = Form("auto"),
    ):
        try:
            language = pipeline.check_language(language)  # cheapest check first
        except UnsupportedLanguageError as exc:
            raise HTTPException(400, str(exc)) from exc

        path = _save_upload(file, max_bytes)  # body already size-capped by BodySizeLimit
        try:
            info = probe(path)  # validate by content, before queueing any work
        except AudioDecodeError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(415, f"unsupported or corrupt audio: {exc}") from exc
        if info.duration is not None and info.duration > config.max_duration_seconds:
            path.unlink(missing_ok=True)
            raise HTTPException(413, f"audio longer than {config.max_duration_seconds:.0f}s")

        # Sync is a convenience for short clips, never a way to hold a request open for an
        # hour: an explicit sync request for long (or unknown-length) audio is refused.
        short = info.duration is not None and info.duration <= config.sync_max_seconds
        if mode == "sync" and not short:
            path.unlink(missing_ok=True)
            raise HTTPException(
                400, f"mode=sync is limited to {config.sync_max_seconds:.0f}s of audio; use mode=async"
            )

        try:
            job = jobs.create(file.filename or "upload", language)
        except QueueFullError:
            path.unlink(missing_ok=True)
            raise HTTPException(429, "transcription queue is full, retry later",
                                headers={"Retry-After": "30"}) from None
        executor.submit(run_job, job, path)

        def accepted() -> JSONResponse:
            return JSONResponse(
                {**job.summary(), "status_url": f"/v1/jobs/{job.id}"},
                status_code=202,
                headers={"Location": f"/v1/jobs/{job.id}"},
            )

        if mode == "async" or not short:
            return accepted()
        # A short clip can still sit behind long jobs in the queue. Never wait unboundedly:
        # past the deadline the job keeps running and the client gets a pollable 202.
        if not job.done.wait(timeout=config.sync_wait_seconds):
            return accepted()
        if job.status is Status.failed:
            raise HTTPException(422, job.error)
        return respond(job.transcript, format)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, format: Format = Query("json")):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        if job.status is not Status.succeeded:
            return job.summary()
        if format == "json":
            return {**job.summary(), "transcript": job.transcript.model_dump()}
        return respond(job.transcript, format)

    return app


def _save_upload(file: UploadFile, max_bytes: int) -> Path:
    """Stream the upload to disk in 1 MiB blocks, enforcing the size cap as we go."""
    suffix = Path(file.filename or "").suffix[:10]  # hint only; we decode by content
    fd, name = tempfile.mkstemp(prefix="upload-", suffix=suffix)
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while block := file.file.read(1 << 20):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(413, f"upload exceeds {max_bytes // (1 << 20)} MB")
                out.write(block)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    if written == 0:
        Path(name).unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    return Path(name)

