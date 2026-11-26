"""CLI: transcribe a local file.

    transcribe meeting.mp3                       # JSON to stdout
    transcribe meeting.mp3 -f srt -o meeting.srt
    transcribe podcast.m4a --model small --language en --chunk-seconds 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transcriber.audio import AudioDecodeError
from transcriber.config import PipelineConfig
from transcriber.formats import FORMATS, render
from transcriber.pipeline import AudioTooLongError, TranscriptionPipeline, UnsupportedLanguageError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="transcribe", description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path)
    p.add_argument("-f", "--format", choices=FORMATS, default="json")
    p.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    p.add_argument("--language", help="ISO code, e.g. en; omit to auto-detect")
    p.add_argument("--engine", choices=["whisper", "mock"])
    p.add_argument("--model", dest="model_size", help="tiny | base | small | medium | large-v3")
    p.add_argument("--device", help="cpu | cuda | auto")
    p.add_argument("--chunk-seconds", type=float)
    p.add_argument("--vad", choices=["auto", "on", "off"],
                   help="speech filter: auto (default) falls back to off for music/singing")
    p.add_argument("--hotwords", help='domain terms to bias toward, e.g. "DORA ICT"')
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if not args.input.is_file():
        p.error(f"no such file: {args.input}")

    config = PipelineConfig.from_env(
        engine=args.engine,
        model_size=args.model_size,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        hotwords=args.hotwords,
        vad=args.vad,
    )

    def progress(done: float, total: float | None) -> None:
        if total:
            print(f"\r{done:7.1f}s / {total:.1f}s ({100 * min(done / total, 1):5.1f}%)",
                  end="", file=sys.stderr, flush=True)

    try:
        transcript = TranscriptionPipeline(config).run(args.input, args.language, on_progress=progress)
    except (AudioDecodeError, AudioTooLongError, UnsupportedLanguageError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    print(file=sys.stderr)
    for w in transcript.warnings:
        print(f"warning: {w}", file=sys.stderr)

    out = render(transcript, args.format)
    if args.output:
        args.output.write_text(out, encoding="utf-8")
        print(f"wrote {args.output} ({len(transcript.segments)} segments, "
              f"{transcript.duration:.1f}s audio, RTF {transcript.real_time_factor})", file=sys.stderr)
    else:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
