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
