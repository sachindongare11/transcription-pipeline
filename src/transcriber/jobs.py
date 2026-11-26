"""Job registry for asynchronous transcriptions.

`InMemoryJobStore` keeps this runnable as a single command. Its limits are
deliberate and explicit rather than accidental:

  * finished jobs (and their transcripts) are evicted after `ttl_seconds`, so
    memory stays bounded however long the process runs;
  * state is per process: run the API with ONE worker process. More uvicorn
    workers or replicas would each hold their own jobs, and a status poll that
    lands on another process returns 404; a restart loses queued jobs.

Scaling out replaces this class, not the API: same `create/get` contract backed
by Redis or a database, with jobs dispatched through a queue to GPU workers.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from transcriber.schemas import Transcript


class Status(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class QueueFullError(Exception):
    pass


@dataclass
class Job:
    id: str
    filename: str
    language: str | None
    status: Status = Status.queued
    progress: float = 0.0
    transcript: Transcript | None = None
    error: str | None = None
    finished_at: float | None = None  # monotonic clock
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def active(self) -> bool:
        return self.status in (Status.queued, Status.running)

    def finish(self, status: Status, error: str | None = None) -> None:
        self.status, self.error = status, error
        self.finished_at = time.monotonic()
        self.done.set()

    def summary(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "filename": self.filename,
            "progress": round(self.progress, 3),
            "error": self.error,
        }


class InMemoryJobStore:
    def __init__(self, ttl_seconds: float, max_active: int):
        self.ttl_seconds = ttl_seconds
        self.max_active = max_active
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, language: str | None) -> Job:
        with self._lock:
            self._evict_expired()
            if sum(j.active for j in self._jobs.values()) >= self.max_active:
                raise QueueFullError
            job = Job(uuid.uuid4().hex, filename, language)
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._evict_expired()
            return self._jobs.get(job_id)

    def __len__(self) -> int:
        return len(self._jobs)

    def _evict_expired(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        expired = [k for k, j in self._jobs.items() if j.finished_at is not None and j.finished_at < cutoff]
        for k in expired:
            del self._jobs[k]
