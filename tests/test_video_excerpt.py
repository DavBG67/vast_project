from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vast_upscaler.video import run_ffprobe
from vast_upscaler.video_excerpt import (
    VideoExcerptError,
    create_video_excerpt,
    parse_timecode_to_seconds,
    resolve_excerpt_window,
)

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run_cmd(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _require_ffmpeg_encoder(encoder: str) -> None:
    listing = _run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-encoders"])
    if listing.returncode != 0 or encoder not in listing.stdout:
        pytest.skip(f"ffmpeg encoder '{encoder}' is unavailable in this environment")


def test_parse_timecode_to_seconds_accepts_mmss_and_hhmmss() -> None:
    assert parse_timecode_to_seconds("00:00") == 0
    assert parse_timecode_to_seconds("01:20") == 80
    assert parse_timecode_to_seconds("01:02:03") == 3723


def test_parse_timecode_to_seconds_rejects_invalid_values() -> None:
    with pytest.raises(VideoExcerptError):
        parse_timecode_to_seconds("abc")
    with pytest.raises(VideoExcerptError):
        parse_timecode_to_seconds("12:61")
    with pytest.raises(VideoExcerptError):
        parse_timecode_to_seconds("01:61:00")


def test_resolve_excerpt_window_rejects_out_of_bounds_range() -> None:
    with pytest.raises(VideoExcerptError):
        resolve_excerpt_window(start_timecode="00:10", requested_duration_seconds=5, source_duration_seconds=12)


@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe is required for excerpt integration tests")
def test_create_video_excerpt_preserves_audio_video_streams(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    excerpt = tmp_path / "excerpt.mkv"
    build = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-t",
            "3.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        raise AssertionError(build.stderr)

    result = create_video_excerpt(
        source_path=source,
        excerpt_path=excerpt,
        start_seconds=1.0,
        duration_seconds=1.2,
    )
    assert excerpt.exists()
    assert 0.8 <= result.actual_duration_seconds <= 1.4
    assert result.actual_frame_count > 0

    payload = run_ffprobe(excerpt)
    streams = payload.get("streams", [])
    audio_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"]
    video_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"]
    assert len(video_streams) == 1
    assert len(audio_streams) >= 1


@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe is required for excerpt integration tests")
@pytest.mark.parametrize(
    ("label", "source_name", "video_encoder", "audio_encoder", "fmt_name", "expected_video_codec", "expected_audio_codec"),
    [
        ("webm-excerpt", "Source Clip.WEBM", "libvpx-vp9", "libopus", "webm", "vp9", "opus"),
        ("mkv-excerpt", "Source Clip.MKV", "libx265", "aac", "matroska", "hevc", "aac"),
    ],
)
def test_create_video_excerpt_on_webm_and_mkv_sources(
    tmp_path: Path,
    label: str,
    source_name: str,
    video_encoder: str,
    audio_encoder: str,
    fmt_name: str,
    expected_video_codec: str,
    expected_audio_codec: str,
) -> None:
    _require_ffmpeg_encoder(video_encoder)
    _require_ffmpeg_encoder(audio_encoder)
    _require_ffmpeg_encoder("libx264")
    _require_ffmpeg_encoder("aac")

    case_dir = tmp_path / label / "with spaces"
    case_dir.mkdir(parents=True)
    source = case_dir / source_name
    excerpt = case_dir / "Excerpt Output.MKV"
    source_duration = 3.0
    expected_fps = 24.0
    build = _run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=352x198:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            str(source_duration),
            "-c:v",
            video_encoder,
            "-c:a",
            audio_encoder,
            "-f",
            fmt_name,
            str(source),
        ]
    )
    if build.returncode != 0:
        pytest.skip(f"Cannot generate source '{label}' with available ffmpeg codecs: {build.stderr.strip()}")

    source_payload = run_ffprobe(source)
    source_streams = source_payload.get("streams", [])
    source_video = [s for s in source_streams if isinstance(s, dict) and s.get("codec_type") == "video"]
    source_audio = [s for s in source_streams if isinstance(s, dict) and s.get("codec_type") == "audio"]
    assert len(source_video) == 1
    assert len(source_audio) == 1
    assert source_video[0].get("codec_name") == expected_video_codec
    assert source_audio[0].get("codec_name") == expected_audio_codec

    result = create_video_excerpt(
        source_path=source,
        excerpt_path=excerpt,
        start_seconds=0.7,
        duration_seconds=1.3,
    )
    assert excerpt.exists()
    assert 1.0 <= result.actual_duration_seconds <= 1.6
    assert result.actual_frame_count > 0

    excerpt_payload = run_ffprobe(excerpt)
    excerpt_streams = excerpt_payload.get("streams", [])
    excerpt_video = [s for s in excerpt_streams if isinstance(s, dict) and s.get("codec_type") == "video"]
    excerpt_audio = [s for s in excerpt_streams if isinstance(s, dict) and s.get("codec_type") == "audio"]
    assert len(excerpt_video) == 1
    assert len(excerpt_audio) == 1
    assert excerpt_video[0].get("codec_name") == "h264"
    assert excerpt_audio[0].get("codec_name") == "aac"

    fps_raw = excerpt_video[0].get("avg_frame_rate")
    assert isinstance(fps_raw, str) and "/" in fps_raw
    num_str, den_str = fps_raw.split("/", maxsplit=1)
    excerpt_fps = float(num_str) / float(den_str)
    assert abs(excerpt_fps - expected_fps) <= 0.2

    excerpt_duration = float(excerpt_payload["format"]["duration"])
    assert abs(excerpt_duration - 1.3) <= 0.25

    audio_start = float(excerpt_audio[0].get("start_time") or 0.0)
    video_start = float(excerpt_video[0].get("start_time") or 0.0)
    assert abs(audio_start - video_start) <= 0.1
