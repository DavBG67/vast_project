from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vast_upscaler.video import VideoProbeError, run_ffprobe


class SourceAnalysisError(RuntimeError):
    pass


class AudioTrackInfo(BaseModel):
    index: int
    codec: str | None = None
    language: str | None = None
    channels: int | None = None
    bitrate_bps: int | None = None


class SubtitleTrackInfo(BaseModel):
    index: int
    codec: str | None = None
    language: str | None = None


class InterlaceInfo(BaseModel):
    field_order: str | None = None
    interlaced: bool | None = None
    confidence: Literal["certain", "uncertain"]
    note: str


class SourceVideoAnalysis(BaseModel):
    file_path: str
    size_bytes: int | None = None
    duration_seconds: float | None = None
    container_format: str | None = None
    overall_bitrate_bps: int | None = None
    video_codec: str | None = None
    video_bitrate_bps: int | None = None
    width: int | None = None
    height: int | None = None
    frame_rate_fps: float | None = None
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    pixel_format: str | None = None
    audio_stream_count: int = 0
    subtitle_stream_count: int = 0
    audio_tracks: list[AudioTrackInfo]
    subtitle_tracks: list[SubtitleTrackInfo]
    interlace: InterlaceInfo


def analyze_source_video(video_path: Path, ffprobe_bin: str = "ffprobe") -> SourceVideoAnalysis:
    try:
        payload = run_ffprobe(video_path=video_path, ffprobe_bin=ffprobe_bin)
    except VideoProbeError as exc:
        raise SourceAnalysisError(str(exc)) from exc
    return parse_source_probe(payload=payload, video_path=video_path)


def parse_source_probe(*, payload: dict, video_path: Path) -> SourceVideoAnalysis:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise SourceAnalysisError("ffprobe payload missing streams list")

    fmt = payload.get("format")
    if not isinstance(fmt, dict):
        fmt = {}

    video_stream = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        raise SourceAnalysisError("No video stream found in input")

    audio_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"]
    subtitle_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "subtitle"]
    interlace = _resolve_interlace_info(video_stream)

    return SourceVideoAnalysis(
        file_path=str(video_path),
        size_bytes=_to_int(fmt.get("size")),
        duration_seconds=_to_float(fmt.get("duration")),
        container_format=_to_str(fmt.get("format_name")),
        overall_bitrate_bps=_to_int(fmt.get("bit_rate")),
        video_codec=_to_str(video_stream.get("codec_name")),
        video_bitrate_bps=_to_int(video_stream.get("bit_rate")),
        width=_to_int(video_stream.get("width")),
        height=_to_int(video_stream.get("height")),
        frame_rate_fps=_parse_rate(video_stream.get("avg_frame_rate")),
        sample_aspect_ratio=_to_str(video_stream.get("sample_aspect_ratio")),
        display_aspect_ratio=_to_str(video_stream.get("display_aspect_ratio")),
        pixel_format=_to_str(video_stream.get("pix_fmt")),
        audio_stream_count=len(audio_streams),
        subtitle_stream_count=len(subtitle_streams),
        audio_tracks=[
            AudioTrackInfo(
                index=_to_int(item.get("index")) or index,
                codec=_to_str(item.get("codec_name")),
                language=_stream_language(item),
                channels=_to_int(item.get("channels")),
                bitrate_bps=_to_int(item.get("bit_rate")),
            )
            for index, item in enumerate(audio_streams)
        ],
        subtitle_tracks=[
            SubtitleTrackInfo(
                index=_to_int(item.get("index")) or index,
                codec=_to_str(item.get("codec_name")),
                language=_stream_language(item),
            )
            for index, item in enumerate(subtitle_streams)
        ],
        interlace=interlace,
    )


def _resolve_interlace_info(video_stream: dict) -> InterlaceInfo:
    field_order = _to_str(video_stream.get("field_order"))
    if field_order is None:
        return InterlaceInfo(
            field_order=None,
            interlaced=None,
            confidence="uncertain",
            note="ffprobe did not provide field_order.",
        )

    normalized = field_order.lower()
    if normalized in {"progressive", "unknown"}:
        return InterlaceInfo(
            field_order=field_order,
            interlaced=False if normalized == "progressive" else None,
            confidence="certain" if normalized == "progressive" else "uncertain",
            note="Interlacing inferred from ffprobe field_order.",
        )

    return InterlaceInfo(
        field_order=field_order,
        interlaced=True,
        confidence="certain",
        note="Interlacing inferred from ffprobe field_order.",
    )


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


def _stream_language(stream: dict) -> str | None:
    tags = stream.get("tags")
    if not isinstance(tags, dict):
        return None
    value = tags.get("language")
    if isinstance(value, str) and value:
        return value
    return None
