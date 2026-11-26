import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from transcriber.api import create_app
from transcriber.config import PipelineConfig
from transcriber.engines import MockEngine
from transcriber.jobs import InMemoryJobStore, QueueFullError, Status
from transcriber.limits import BodySizeLimit

MB = 1 << 20


def drive(middleware, chunks, headers=()):
    """Run the middleware against a fake client streaming `chunks`; record what happened."""
    pulled, sent = [], []

    async def receive():
        i = len(pulled)
        pulled.append(i)
        body = chunks[i] if i < len(chunks) else b""
        return {"type": "http.request", "body": body, "more_body": i + 1 < len(chunks)}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": list(headers)}
    asyncio.run(middleware(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, len(pulled)


async def read_everything_app(scope, receive, send):
    while (await receive()).get("more_body"):
        pass
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_declared_oversize_body_is_rejected_before_reading():
    mw = BodySizeLimit(read_everything_app, max_bytes=MB)
    status, pulled = drive(mw, [b"x" * MB] * 50, headers=[(b"content-length", str(50 * MB).encode())])
    assert status == 413 and pulled == 0


def test_undeclared_chunked_body_is_cut_off_at_the_limit():
    mw = BodySizeLimit(read_everything_app, max_bytes=3 * MB)
    status, pulled = drive(mw, [b"x" * MB] * 1000)  # a "1 GB" upload with no Content-Length
    assert status == 413 and pulled == 4  # stopped as soon as the 4th MB crossed the limit


def test_body_under_the_limit_passes_through():
    mw = BodySizeLimit(read_everything_app, max_bytes=3 * MB)
    assert drive(mw, [b"x" * MB] * 2) == (200, 2)


def test_api_reports_413_not_a_parse_error_for_chunked_oversize_upload():
    app = create_app(PipelineConfig(engine="mock", max_upload_mb=1), MockEngine())
    with TestClient(app) as client:
        r = client.post(
            "/v1/transcriptions",
            content=iter([b"x" * MB] * 4),  # generator -> chunked transfer, no Content-Length
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
    assert r.status_code == 413 and "exceeds" in r.json()["detail"]


def test_finished_jobs_expire_after_ttl_but_active_ones_never_do():
    store = InMemoryJobStore(ttl_seconds=60, max_active=10)
    done, running = store.create("a.wav", None), store.create("b.wav", None)
    running.status = Status.running
    done.finish(Status.succeeded)
    assert store.get(done.id) is done

    done.finished_at = time.monotonic() - 61  # pretend a minute passed
    assert store.get(done.id) is None
    assert store.get(running.id) is running and len(store) == 1


def test_queue_limit_counts_only_active_jobs():
    store = InMemoryJobStore(ttl_seconds=60, max_active=1)
    first = store.create("a.wav", None)
    with pytest.raises(QueueFullError):
        store.create("b.wav", None)
    first.finish(Status.failed, "boom")
    assert store.create("b.wav", None)  # finished jobs don't hold a slot
