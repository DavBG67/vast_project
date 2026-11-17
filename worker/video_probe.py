from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class StreamSummary:
    width: int
    height: int
    fps: float
    sar: str | None
    dar: str | None
    codec: str | None
    duration_seconds: float | None
    interlaced: bool | None
    audio_streams: int
    subtitle_streams: int


def run_ffprobe(input_path: Path, ffprobe_bin: str = "ffprobe") -> dict:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ProbeError(f"ffprobe execution failed: {exc}") from exc

    if result.returncode != 0:
        raise ProbeError(result.stderr.strip() or "ffprobe returned a non-zero exit code")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise ProbeError("ffprobe payload root is not an object")
    return payload


def parse_probe(payload: dict) -> StreamSummary:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ProbeError("ffprobe payload missing streams list")

    video_stream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        raise ProbeError("No video stream found")

    width = _to_int(video_stream.get("width"))
    height = _to_int(video_stream.get("height"))
    if width is None or height is None:
        raise ProbeError("Missing video dimensions")

    fps = _parse_rate(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_rate(video_stream.get("r_frame_rate"))
    if fps is None:
        raise ProbeError("Unable to determine frame rate")

    format_obj = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    duration = _to_float(format_obj.get("duration"))

    field_order = str(video_stream.get("field_order") or "").lower()
    interlaced = None
    if field_order:
        interlaced = field_order not in {"progressive", "unknown"}

    audio_streams = sum(1 for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio")
    subtitle_streams = sum(1 for s in streams if isinstance(s, dict) and s.get("codec_type") == "subtitle")

    return StreamSummary(
        width=width,
        height=height,
        fps=fps,
        sar=_to_str(video_stream.get("sample_aspect_ratio")),
        dar=_to_str(video_stream.get("display_aspect_ratio")),
        codec=_to_str(video_stream.get("codec_name")),
        duration_seconds=duration,
        interlaced=interlaced,
        audio_streams=audio_streams,
        subtitle_streams=subtitle_streams,
    )


def probe_video(input_path: Path, ffprobe_bin: str = "ffprobe") -> StreamSummary:
    return parse_probe(run_ffprobe(input_path=input_path, ffprobe_bin=ffprobe_bin))


def _parse_rate(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if text in {"", "0/0", "N/A"}:
        return None
    if "/" in text:
        left, right = text.split("/", maxsplit=1)
        num = _to_float(left)
        den = _to_float(right)
        if num is None or den in (None, 0.0):
            return None
        return num / den
    return _to_float(text)


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
