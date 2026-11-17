from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel


class VideoProbeError(RuntimeError):
    pass


class VideoInfo(BaseModel):
    file_path: str
    size_bytes: int | None = None
    duration_seconds: float | None = None
    container_format: str | None = None
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None
    frame_rate_fps: float | None = None
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    interlaced: bool | None = None
    pixel_format: str | None = None
    audio_codec: str | None = None
    audio_stream_count: int = 0
    subtitle_stream_count: int = 0


def _parse_frame_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num_str, den_str = value.split("/", maxsplit=1)
        num = float(num_str)
        den = float(den_str)
        if den == 0:
            return None
        return num / den
    return float(value)


def run_ffprobe(video_path: Path, ffprobe_bin: str = "ffprobe") -> dict:
    if not video_path.exists():
        raise VideoProbeError(f"Input file does not exist: {video_path}")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(video_path),
    ]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoProbeError(f"ffprobe execution failed: {exc}") from exc

    if completed.returncode != 0:
        raise VideoProbeError(completed.stderr.strip() or "ffprobe returned a non-zero exit code")

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError("ffprobe returned invalid JSON") from exc


def parse_ffprobe_data(data: dict, video_path: Path) -> VideoInfo:
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

    if video_stream is None:
        raise VideoProbeError("No video stream found in input")

    field_order = (video_stream.get("field_order") or "").lower()
    interlaced = None
    if field_order:
        interlaced = field_order not in {"progressive", "unknown"}

    return VideoInfo(
        file_path=str(video_path),
        size_bytes=int(fmt["size"]) if fmt.get("size") else None,
        duration_seconds=float(fmt["duration"]) if fmt.get("duration") else None,
        container_format=fmt.get("format_name"),
        video_codec=video_stream.get("codec_name"),
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        frame_rate_fps=_parse_frame_rate(video_stream.get("avg_frame_rate")),
        sample_aspect_ratio=video_stream.get("sample_aspect_ratio"),
        display_aspect_ratio=video_stream.get("display_aspect_ratio"),
        interlaced=interlaced,
        pixel_format=video_stream.get("pix_fmt"),
        audio_codec=audio_streams[0].get("codec_name") if audio_streams else None,
        audio_stream_count=len(audio_streams),
        subtitle_stream_count=len(subtitle_streams),
    )


def inspect_video(video_path: Path, ffprobe_bin: str = "ffprobe") -> VideoInfo:
    data = run_ffprobe(video_path=video_path, ffprobe_bin=ffprobe_bin)
    return parse_ffprobe_data(data=data, video_path=video_path)
