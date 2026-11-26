"""Output contract for downstream consumers (search indexing, summarisation, subtitles)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float = Field(ge=0.0, le=1.0)


class Segment(BaseModel):
    id: int
    start: float = Field(description="seconds from start of source audio")
    end: float
    text: str
    confidence: float = Field(ge=0.0, le=1.0, description="exp(avg_logprob); ~ per-token probability")
    low_confidence: bool = False
    words: list[Word] = Field(default_factory=list, description="word-level timing, for search/redaction/highlighting")


class SourceInfo(BaseModel):
    filename: str | None = None
    container: str
    codec: str
    sample_rate: int
    channels: int


class Transcript(BaseModel):
    text: str
    language: str | None
    duration: float
    segments: list[Segment]
    source: SourceInfo
    engine: str
    chunks: int
    processing_seconds: float
    real_time_factor: float = Field(description="processing time / audio duration; < 1 is faster than real time")
    warnings: list[str] = Field(default_factory=list, description="why output may be empty or less reliable")
