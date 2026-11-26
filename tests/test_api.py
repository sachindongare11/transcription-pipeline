import threading
import time

import pytest
from fastapi.testclient import TestClient

from transcriber.api import create_app
from transcriber.config import PipelineConfig
from transcriber.engines import MockEngine
from conftest import bursts


@pytest.fixture
def client():
    cfg = PipelineConfig(engine="mock", sync_max_seconds=10, chunk_seconds=5, max_upload_mb=1)
    with TestClient(create_app(cfg, MockEngine())) as c:
        yield c


def upload(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/v1/transcriptions", files={"file": (path.name, f)}, data=form)


def test_short_file_is_transcribed_synchronously(client, make_audio):
    path = make_audio(bursts([(1, 2), (3, 4)], 5.0), "short.mp3", "mp3", "libmp3lame")
    r = upload(client, path)
    assert r.status_code == 200
    body = r.json()
    assert [(round(s["start"]), round(s["end"])) for s in body["segments"]] == [(1, 2), (3, 4)]
    assert body["source"]["filename"] == "short.mp3"


def test_subtitle_format(client, make_audio):
    r = upload(client, make_audio(bursts([(1, 2)], 3.0)), format="vtt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/vtt")
    assert r.text.startswith("WEBVTT")


def test_long_file_becomes_async_job(client, make_audio):
    path = make_audio(bursts([(t, t + 1) for t in range(0, 20, 2)], 20.0), "long.ogg", "ogg", "libopus", rate=48000)
    r = upload(client, path)
    assert r.status_code == 202
    status_url = r.json()["status_url"]
    assert r.headers["location"] == status_url

    for _ in range(100):
        job = client.get(status_url).json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded" and job["progress"] == 1.0
    assert len(job["transcript"]["segments"]) == 10
    assert client.get(status_url, params={"format": "srt"}).text.startswith("1\n00:00:00")


@pytest.mark.parametrize(
    "content,status",
    [(b"definitely not audio", 415), (b"", 400), (b"\0" * (2 << 20), 413)],
)
def test_bad_uploads_are_rejected(client, tmp_path, content, status):
    path = tmp_path / "x.wav"
    path.write_bytes(content)
    assert upload(client, path).status_code == status


def test_unknown_job(client):
    assert client.get("/v1/jobs/nope").status_code == 404


class SlowEngine(MockEngine):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def transcribe(self, pcm, language=None, vad=True):
        time.sleep(self.delay)
        return super().transcribe(pcm, language, vad)


def wait_for(client, url):
    for _ in range(200):
        job = client.get(url).json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_explicit_sync_is_refused_for_long_audio(client, make_audio):
    path = make_audio(bursts([(1, 2)], 30.0))  # > sync_max_seconds=10 in the fixture
    r = upload(client, path, mode="sync")
    assert r.status_code == 400 and "mode=async" in r.json()["detail"]


def test_sync_request_stops_waiting_and_hands_back_a_job(make_audio):
    cfg = PipelineConfig(engine="mock", sync_max_seconds=10, sync_wait_seconds=0.2)
    with TestClient(create_app(cfg, SlowEngine(delay=1.0))) as c:
        t0 = time.monotonic()
        r = upload(c, make_audio(bursts([(1, 2)], 3.0)), mode="sync")
        assert r.status_code == 202 and time.monotonic() - t0 < 0.9
        assert wait_for(c, r.json()["status_url"])["status"] == "succeeded"


def test_language_is_validated_and_normalised(make_audio):
    class EnglishOnly(MockEngine):
        languages = frozenset({"en"})

    with TestClient(create_app(PipelineConfig(engine="mock"), EnglishOnly())) as c:
        path = make_audio(bursts([(1, 2)], 3.0))
        bad = upload(c, path, language="xx")
        assert bad.status_code == 400 and "unsupported language" in bad.json()["detail"]
        ok = upload(c, path, language=" EN ")
        assert ok.status_code == 200 and ok.json()["language"] == "en"


def test_decode_errors_do_not_leak_server_paths(client, tmp_path):
    path = tmp_path / "x.mp3"
    path.write_bytes(b"not audio")
    detail = upload(client, path).json()["detail"]
    assert "upload-" not in detail and "/" not in detail


def test_readiness_waits_for_the_model():
    gate = threading.Event()

    class SlowLoad(MockEngine):
        def load(self):
            gate.wait(5)

    with TestClient(create_app(PipelineConfig(engine="mock"), SlowLoad())) as c:
        assert c.get("/healthz").status_code == 200  # alive...
        assert c.get("/readyz").status_code == 503   # ...but not ready
        gate.set()
        for _ in range(100):
            if c.get("/readyz").status_code == 200:
                break
            time.sleep(0.01)
        assert c.get("/readyz").json()["ready"] is True
