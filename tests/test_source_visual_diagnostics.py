from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vast_upscaler.source_analysis import (
    SourceAnalysisError,
    VisualDiagnosticsOptions,
    analyze_source_video_report,
)

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg is required for visual diagnostics tests")


def _run_ffmpeg(cmd: list[str]) -> None:
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise AssertionError(f"ffmpeg failed ({run.returncode}): {run.stderr}")


def _make_test_video(path: Path, vf: str, *, duration: float = 1.2) -> Path:
    _run_ffmpeg(
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
            "-t",
            f"{duration:.3f}",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


def _make_dark_video(path: Path, *, duration: float = 1.2) -> Path:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=320x180:rate=25",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


def _make_interlaced_like_video(path: Path, *, duration: float = 1.2) -> Path:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25",
            "-t",
            f"{duration:.3f}",
            "-vf",
            "tinterlace=interleave_top",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


def _options() -> VisualDiagnosticsOptions:
    return VisualDiagnosticsOptions(
        sample_count=3,
        sample_duration_seconds=0.4,
        max_analyzed_duration_seconds=4.0,
        ffmpeg_threads=1,
        ffmpeg_timeout_seconds=30.0,
        min_samples_required=2,
    )


def test_visual_diagnostics_progressive_without_black_bars(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "progressive.mp4", "scale=320:180")
    payload = analyze_source_video_report(
        video_path=video,
        include_visual_diagnostics=True,
        diagnostics_options=_options(),
    )
    report = payload["analysis_report"]["visual_diagnostics"]
    crop = report["cropdetect"]
    assert crop["status"] in {"ok", "partial"}
    margins = crop.get("probable_margins") or {}
    assert margins.get("top", 0) <= 4
    assert margins.get("bottom", 0) <= 4
    assert margins.get("left", 0) <= 4
    assert margins.get("right", 0) <= 4


def test_visual_diagnostics_detects_horizontal_black_bars(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "hbars.mp4", "scale=320:140,pad=320:180:0:20:black")
    payload = analyze_source_video_report(
        video_path=video,
        include_visual_diagnostics=True,
        diagnostics_options=_options(),
    )
    margins = payload["analysis_report"]["visual_diagnostics"]["cropdetect"]["probable_margins"]
    assert margins is not None
    assert margins["top"] >= 12
    assert margins["bottom"] >= 12


def test_visual_diagnostics_detects_vertical_black_bars(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "vbars.mp4", "scale=220:180,pad=320:180:50:0:black")
    payload = analyze_source_video_report(
        video_path=video,
        include_visual_diagnostics=True,
        diagnostics_options=_options(),
    )
    margins = payload["analysis_report"]["visual_diagnostics"]["cropdetect"]["probable_margins"]
    assert margins is not None
    assert margins["left"] >= 30
    assert margins["right"] >= 30


def test_visual_diagnostics_interlace_signal_not_forced_to_progressive(tmp_path: Path) -> None:
    video = _make_interlaced_like_video(tmp_path / "interlaced.mp4")
    payload = analyze_source_video_report(
        video_path=video,
        include_visual_diagnostics=True,
        diagnostics_options=_options(),
    )
    classification = payload["analysis_report"]["visual_diagnostics"]["idet"]["classification"]
    assert classification in {"interlaced", "mixed", "indeterminate"}


def test_visual_diagnostics_dark_video_marks_low_confidence_noise(tmp_path: Path) -> None:
    video = _make_dark_video(tmp_path / "dark.mp4")
    payload = analyze_source_video_report(
        video_path=video,
        include_visual_diagnostics=True,
        diagnostics_options=_options(),
    )
    noise = payload["analysis_report"]["visual_diagnostics"]["noise_artifacts"]
    assert noise["confidence"] in {"low", "unknown"}
    notes = " ".join(noise["notes"]).lower()
    assert "dark" in notes


def test_visual_diagnostics_short_video_reports_limited_confidence(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "short.mp4", "scale=320:180", duration=0.2)
    payload = analyze_source_video_report(
        video_path=video,
        include_visual_diagnostics=True,
        diagnostics_options=_options(),
    )
    idet = payload["analysis_report"]["visual_diagnostics"]["idet"]
    assert idet["confidence"] in {"low", "unknown"}
    assert idet["analyzed_samples"] >= 1


def test_visual_diagnostics_returns_partial_when_ffmpeg_filter_execution_fails(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "sample.mp4", "scale=320:180")
    payload = analyze_source_video_report(
        video_path=video,
        include_visual_diagnostics=True,
        diagnostics_options=_options(),
        ffmpeg_bin="ffmpeg-not-found",
    )
    report = payload["analysis_report"]["visual_diagnostics"]
    assert report["partial"] is True
    assert report["cropdetect"]["status"] == "failed"
    assert report["idet"]["status"] == "failed"


def test_visual_diagnostics_invalid_file_raises_clean_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.mp4"
    invalid.write_text("not a video", encoding="utf-8")
    with pytest.raises(SourceAnalysisError):
        analyze_source_video_report(video_path=invalid, include_visual_diagnostics=True, diagnostics_options=_options())
