from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vast_upscaler.video import VideoProbeError, inspect_video, run_ffprobe


class VideoExcerptError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoExcerptResult:
    source_path: Path
    excerpt_path: Path
    start_seconds: float
    requested_duration_seconds: float
    actual_duration_seconds: float
    actual_frame_count: int
    frame_rate_fps: float | None
    source_duration_seconds: float


_TIMECODE_PATTERN = re.compile(r"^\d{1,3}:\d{2}(?::\d{2})?$")


def parse_timecode_to_seconds(value: str) -> float:
    text = value.strip()
    if not text:
        raise VideoExcerptError("Excerpt start time is required (MM:SS or HH:MM:SS).")
    if not _TIMECODE_PATTERN.fullmatch(text):
        raise VideoExcerptError("Invalid excerpt start format. Use MM:SS or HH:MM:SS.")
    parts = [int(item) for item in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        if seconds >= 60:
            raise VideoExcerptError("Invalid excerpt start format. Seconds must be < 60.")
        return float((minutes * 60) + seconds)

    hours, minutes, seconds = parts
    if minutes >= 60 or seconds >= 60:
        raise VideoExcerptError("Invalid excerpt start format. Minutes/seconds must be < 60 in HH:MM:SS.")
    return float((hours * 3600) + (minutes * 60) + seconds)


def resolve_excerpt_window(
    *,
    start_timecode: str | None,
    requested_duration_seconds: float | None,
    source_duration_seconds: float | None,
) -> tuple[float, float]:
    if source_duration_seconds is None or source_duration_seconds <= 0:
        raise VideoExcerptError("Source duration is unavailable; cannot validate excerpt range.")
    start_seconds = parse_timecode_to_seconds(start_timecode or "00:00")
    duration_seconds = float(requested_duration_seconds if requested_duration_seconds is not None else 10.0)
    if duration_seconds <= 0:
        raise VideoExcerptError("Excerpt duration must be strictly positive.")
    if start_seconds < 0:
        raise VideoExcerptError("Excerpt start must be >= 0.")
    if start_seconds >= source_duration_seconds:
        raise VideoExcerptError("Excerpt start is outside source duration.")
    max_duration = source_duration_seconds - start_seconds
    if duration_seconds > max_duration + 0.001:
        raise VideoExcerptError("Excerpt range exceeds source duration.")
    return start_seconds, duration_seconds


def create_video_excerpt(
    *,
    source_path: Path,
    excerpt_path: Path,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> VideoExcerptResult:
    if duration_seconds <= 0:
        raise VideoExcerptError("Excerpt duration must be strictly positive.")
    try:
        source_info = inspect_video(source_path, ffprobe_bin=ffprobe_bin)
    except VideoProbeError as exc:
        raise VideoExcerptError(str(exc)) from exc
    source_duration = source_info.duration_seconds
    if source_duration is None or source_duration <= 0:
        raise VideoExcerptError("Source duration is unavailable; cannot prepare excerpt.")
    if start_seconds < 0 or start_seconds >= source_duration:
        raise VideoExcerptError("Excerpt start is outside source duration.")
    if duration_seconds > (source_duration - start_seconds) + 0.001:
        raise VideoExcerptError("Excerpt range exceeds source duration.")

    excerpt_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-c:s",
        "copy",
        str(excerpt_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoExcerptError(f"ffmpeg execution failed while preparing excerpt: {exc}") from exc
    if completed.returncode != 0:
        message = (completed.stderr or "").strip() or "ffmpeg failed while preparing excerpt."
        raise VideoExcerptError(message)

    try:
        payload = run_ffprobe(video_path=excerpt_path, ffprobe_bin=ffprobe_bin)
    except VideoProbeError as exc:
        raise VideoExcerptError(f"Excerpt ffprobe validation failed: {exc}") from exc

    actual_duration = _probe_duration_seconds(payload)
    if actual_duration is None or actual_duration <= 0:
        raise VideoExcerptError("Excerpt validation failed: output duration is unavailable or invalid.")
    frame_rate = _probe_fps(payload)
    frame_count = _probe_frame_count(payload, actual_duration, frame_rate)
    if frame_count is None or frame_count <= 0:
        raise VideoExcerptError("Excerpt validation failed: output frame count is unavailable or invalid.")
    if actual_duration > duration_seconds + 0.25:
        raise VideoExcerptError("Excerpt validation failed: output duration exceeds requested range.")

    return VideoExcerptResult(
        source_path=source_path,
        excerpt_path=excerpt_path,
        start_seconds=start_seconds,
        requested_duration_seconds=duration_seconds,
        actual_duration_seconds=actual_duration,
        actual_frame_count=frame_count,
        frame_rate_fps=frame_rate,
        source_duration_seconds=source_duration,
    )


def _probe_duration_seconds(payload: dict) -> float | None:
    fmt = payload.get("format")
    if not isinstance(fmt, dict):
        return None
    value = fmt.get("duration")
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _probe_fps(payload: dict) -> float | None:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return None
    video_stream = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        return None
    value = video_stream.get("avg_frame_rate")
    if not isinstance(value, str) or value in {"", "0/0", "N/A"}:
        return None
    if "/" in value:
        left, right = value.split("/", maxsplit=1)
        try:
            num = float(left)
            den = float(right)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(value)
    except ValueError:
        return None


def _probe_frame_count(payload: dict, duration_seconds: float, frame_rate_fps: float | None) -> int | None:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return None
    video_stream = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        return None
    nb_frames = video_stream.get("nb_frames")
    try:
        if nb_frames not in (None, "N/A"):
            parsed = int(nb_frames)
            if parsed > 0:
                return parsed
    except (TypeError, ValueError):
        pass
    if frame_rate_fps is None:
        return None
    estimated = round(duration_seconds * frame_rate_fps)
    return estimated if estimated > 0 else None
