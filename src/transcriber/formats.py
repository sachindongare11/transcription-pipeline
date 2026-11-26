"""Render a Transcript into the formats downstream tools expect."""

from __future__ import annotations

from transcriber.schemas import Transcript

FORMATS = ("json", "txt", "srt", "vtt")
MEDIA_TYPES = {
    "json": "application/json",
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip",
    "vtt": "text/vtt; charset=utf-8",
}


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(t: Transcript) -> str:
    blocks = [
        f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{s.text}\n"
        for i, s in enumerate(t.segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    cues = [f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}\n{s.text}\n" for s in t.segments]
    return "WEBVTT\n\n" + "\n".join(cues)


def to_txt(t: Transcript) -> str:
    return "\n".join(f"[{_ts(s.start, '.')} - {_ts(s.end, '.')}] {s.text}" for s in t.segments) + "\n"


def render(t: Transcript, fmt: str) -> str:
    if fmt == "json":
        return t.model_dump_json(indent=2)
    if fmt == "txt":
        return to_txt(t)
    if fmt == "srt":
        return to_srt(t)
    if fmt == "vtt":
        return to_vtt(t)
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
