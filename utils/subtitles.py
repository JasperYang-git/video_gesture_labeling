"""SRT generation and lossless soft-subtitle muxing for predicted gestures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from utils.preprocessing.annotations import AnnotationInterval
from utils.schema import class_id_to_name


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    milliseconds = int(round(seconds * 1000.0))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def segments_to_srt(
    intervals: list[AnnotationInterval],
    show_confidence: bool = True,
) -> str:
    blocks: list[str] = []
    for index, interval in enumerate(intervals, start=1):
        text = class_id_to_name(interval.label)
        if show_confidence:
            text = f"{text} ({interval.confidence:.2f})"
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(interval.start_sec)} --> "
            f"{format_timestamp(interval.end_sec)}\n"
            f"{text}\n"
        )
    return "\n".join(blocks)


def write_srt(
    path: str | Path,
    intervals: list[AnnotationInterval],
    show_confidence: bool = True,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(segments_to_srt(intervals, show_confidence), encoding="utf-8")
    return output


def mux_subtitles(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Attach an SRT as a default-on soft subtitle track without re-encoding."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to mux subtitles; install it or disable the mux option"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(srt_path),
        "-map",
        "0",
        "-map",
        "1",
        "-c",
        "copy",
        "-c:s",
        "mov_text",
        "-disposition:s:0",
        "default",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}: {result.stderr.strip() or result.returncode}"
        )
    return output
