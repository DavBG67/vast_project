from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vast_upscaler.local_preprocessing import (
    LocalExecutionOptions,
    LocalPreprocessingExecutionError,
    LocalPreprocessingValidationError,
    execute_preprocessing_plan,
)
from vast_upscaler.preprocessing_planner import (
    PlannerUserChoices,
    build_preprocessing_plan,
)

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

pytestmark = pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg is required for local preprocessing tests")


def _run_ffmpeg(cmd: list[str]) -> None:
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise AssertionError(f"ffmpeg failed ({run.returncode}): {run.stderr}")


def _make_av_video(path: Path, vf: str, *, duration: float = 1.2) -> Path:
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
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-t",
            f"{duration:.3f}",
            "-vf",
            vf,
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(path),
        ]
    )
    return path


def _analysis_payload(source: Path, *, idet_classification: str = "progressive", idet_confidence: str = "high") -> dict:
    return {
        "file_path": str(source),
        "width": 320,
        "height": 180,
        "frame_rate_fps": 25.0,
        "duration_seconds": 1.2,
        "video_codec": "h264",
        "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
        "analysis_report": {
            "version": "1.0",
            "visual_diagnostics": {
                "partial": False,
                "cropdetect": {
                    "status": "ok",
                    "confidence": "high",
                    "recommended_crop": {"x": 0, "y": 20, "width": 320, "height": 140},
                    "probable_margins": {"top": 20, "bottom": 20, "left": 0, "right": 0},
                },
                "idet": {
                    "status": "ok",
                    "confidence": idet_confidence,
                    "classification": idet_classification,
                },
                "noise_artifacts": {
                    "status": "ok",
                    "confidence": "low",
                    "likely_noise": "unknown",
                    "likely_compression_artifacts": "unknown",
                },
            },
        },
    }


def test_plan_without_enabled_operation_is_rejected(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(analysis_payload=_analysis_payload(source))
    with pytest.raises(LocalPreprocessingValidationError):
        execute_preprocessing_plan(plan=plan, output_path=tmp_path / "out.mp4")


def test_auto_crop_explicit_selection_produces_expected_resolution(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:140,pad=320:180:0:20:black")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    output = tmp_path / "out.mp4"
    report = execute_preprocessing_plan(plan=plan, output_path=output)
    assert output.exists()
    assert report.resolution_after == {"width": 320, "height": 140}
    assert report.status == "completed"


def test_custom_crop_valid_selection_applies(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(
            crop_mode="custom_crop",
            custom_crop_margins={"top": 10, "bottom": 10, "left": 20, "right": 20},
        ),
    )
    output = tmp_path / "custom.mp4"
    report = execute_preprocessing_plan(plan=plan, output_path=output)
    assert report.resolution_after == {"width": 280, "height": 160}


def test_custom_crop_invalid_plan_payload_is_rejected(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    crop = next(op for op in plan.operations if op.operation_id == "crop")
    crop.parameters["crop_rect"] = {"x": 1, "y": 1, "width": 101, "height": 101}
    with pytest.raises(LocalPreprocessingValidationError):
        execute_preprocessing_plan(plan=plan, output_path=tmp_path / "bad.mp4")


def test_explicit_deinterlace_executes(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source, idet_classification="interlaced"),
        user_choices=PlannerUserChoices(deinterlace_mode="enable"),
    )
    output = tmp_path / "deint.mp4"
    report = execute_preprocessing_plan(plan=plan, output_path=output)
    assert "deinterlace" in report.applied_operations
    assert report.resolution_after == {"width": 320, "height": 180}


def test_crop_plus_deinterlace_combo_executes(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:140,pad=320:180:0:20:black")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source, idet_classification="interlaced"),
        user_choices=PlannerUserChoices(crop_mode="auto_crop", deinterlace_mode="enable"),
    )
    output = tmp_path / "combo.mp4"
    report = execute_preprocessing_plan(plan=plan, output_path=output)
    assert report.resolution_after == {"width": 320, "height": 140}
    assert report.applied_operations == ["crop", "deinterlace"]


def test_unsupported_enabled_operation_is_rejected(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop", denoise_mode="enable"),
    )
    with pytest.raises(LocalPreprocessingValidationError):
        execute_preprocessing_plan(plan=plan, output_path=tmp_path / "out.mp4")


def test_output_exists_is_rejected_without_overwrite(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    output = tmp_path / "exists.mp4"
    output.write_bytes(b"already")
    with pytest.raises(LocalPreprocessingValidationError):
        execute_preprocessing_plan(plan=plan, output_path=output)


def test_source_and_destination_must_differ(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    with pytest.raises(LocalPreprocessingValidationError):
        execute_preprocessing_plan(plan=plan, output_path=source)


def test_ffmpeg_error_cleans_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    fixed_temp = tmp_path / "fixed.tmp.mp4"
    monkeypatch.setattr("vast_upscaler.local_preprocessing._build_temp_output_path", lambda _dst: fixed_temp)
    with pytest.raises(LocalPreprocessingExecutionError):
        execute_preprocessing_plan(
            plan=plan,
            output_path=tmp_path / "out.mp4",
            options=LocalExecutionOptions(ffmpeg_bin="ffmpeg-missing", timeout_seconds=5),
        )
    assert not fixed_temp.exists()


def test_audio_stream_is_preserved_when_copy_enabled(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:140,pad=320:180:0:20:black")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    output = tmp_path / "audio-preserved.mp4"
    report = execute_preprocessing_plan(plan=plan, output_path=output)
    assert report.ffprobe_validation is not None
    assert int(report.ffprobe_validation["audio_stream_count"]) >= 1


def test_dry_run_creates_no_output_file(tmp_path: Path) -> None:
    source = _make_av_video(tmp_path / "source.mp4", "scale=320:180")
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(source),
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    output = tmp_path / "dry-run.mp4"
    report = execute_preprocessing_plan(
        plan=plan,
        output_path=output,
        options=LocalExecutionOptions(dry_run=True),
    )
    assert report.status == "dry_run"
    assert not output.exists()
