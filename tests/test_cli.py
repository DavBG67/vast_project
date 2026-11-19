import json
from pathlib import Path

from typer.testing import CliRunner

from vast_upscaler.cli import app
from vast_upscaler.local_preprocessing import LocalExecutionReport
from vast_upscaler.source_analysis import InterlaceInfo, SourceVideoAnalysis
from vast_upscaler.video import VideoInfo

runner = CliRunner()


def test_doctor_success(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("video:\n  target_width: 1920\n", encoding="utf-8")

    monkeypatch.setattr("vast_upscaler.cli.shutil.which", lambda _name: "C:/bin/tool.exe")

    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_inspect_json_output(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "video.mkv"
    video_path.write_bytes(b"x")

    monkeypatch.setattr(
        "vast_upscaler.cli.inspect_video",
        lambda _path: VideoInfo(
            file_path=str(video_path),
            width=720,
            height=576,
            audio_stream_count=1,
            subtitle_stream_count=0,
        ),
    )

    result = runner.invoke(app, ["inspect", str(video_path), "--json"])

    assert result.exit_code == 0
    assert '"width": 720' in result.stdout
    assert '"height": 576' in result.stdout


def test_analyze_source_json_output(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "video.mkv"
    video_path.write_bytes(b"x")

    monkeypatch.setattr(
        "vast_upscaler.cli.analyze_source_video",
        lambda _path: SourceVideoAnalysis(
            file_path=str(video_path),
            width=1280,
            height=720,
            frame_rate_fps=30.0,
            audio_tracks=[],
            subtitle_tracks=[],
            interlace=InterlaceInfo(
                field_order="progressive",
                interlaced=False,
                confidence="certain",
                note="ok",
            ),
        ),
    )

    result = runner.invoke(app, ["analyze-source", str(video_path)])

    assert result.exit_code == 0
    assert '"width": 1280' in result.stdout
    assert '"confidence": "certain"' in result.stdout


def test_analyze_source_visual_diagnostics_option(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "video.mkv"
    video_path.write_bytes(b"x")

    monkeypatch.setattr(
        "vast_upscaler.cli.analyze_source_video_report",
        lambda **_kwargs: {"file_path": str(video_path), "analysis_report": {"version": "1.0"}},
    )

    result = runner.invoke(app, ["analyze-source", str(video_path), "--visual-diagnostics"])

    assert result.exit_code == 0
    assert '"analysis_report"' in result.stdout
    assert '"version": "1.0"' in result.stdout


def test_plan_preprocessing_from_existing_analysis_report_does_not_reanalyze(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "analysis.json"
    report_path.write_text(
        json.dumps(
            {
                "file_path": "sample.mp4",
                "width": 1280,
                "height": 720,
                "frame_rate_fps": 25.0,
                "duration_seconds": 10.0,
                "video_codec": "h264",
                "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
            }
        ),
        encoding="utf-8",
    )

    def _should_not_run(**_kwargs):
        raise AssertionError("analyze_source_video_report must not be called when --analysis-report is provided")

    monkeypatch.setattr("vast_upscaler.cli.analyze_source_video_report", _should_not_run)

    result = runner.invoke(app, ["plan-preprocessing", "--analysis-report", str(report_path)])

    assert result.exit_code == 0
    assert '"version": "1.0"' in result.stdout
    assert '"operations"' in result.stdout


def test_execute_preprocessing_dry_run_command(monkeypatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("vast_upscaler.cli.load_preprocessing_plan", lambda _p: object())
    monkeypatch.setattr(
        "vast_upscaler.cli.execute_preprocessing_plan",
        lambda **_kwargs: LocalExecutionReport(
            status="dry_run",
            source_path="in.mp4",
            destination_path="out.mp4",
            planned_operations=["crop"],
            applied_operations=["crop"],
            filter_chain="crop=100:100:0:0",
            ffmpeg_command=["ffmpeg", "..."],
            encoding={"video_codec": "libx264"},
            resolution_before={"width": 320, "height": 180},
            resolution_after={"width": 320, "height": 140},
            duration_seconds=0.0,
            warnings=[],
            ffprobe_validation=None,
        ),
    )

    result = runner.invoke(
        app,
        ["execute-preprocessing", "--plan", str(plan_path), "--output", str(tmp_path / "out.mp4"), "--dry-run"],
    )

    assert result.exit_code == 0
    assert '"status": "dry_run"' in result.stdout


def test_offers_offline_fixture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
vast:
  gpu_models: [RTX 3090 Ti]
  max_hourly_price_usd: 0.35
  min_reliability: 0.99
  verified_only: true
  disk_gb: 50
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["offers", "--fixture", "tests/fixtures/offers.json", "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert "offline fixture" in result.stdout
    assert "31129" in result.stdout


def test_estimate_offline_fixture(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "video.mkv"
    video_path.write_bytes(b"x")

    monkeypatch.setattr(
        "vast_upscaler.cli.inspect_video",
        lambda _path: VideoInfo(
            file_path=str(video_path),
            width=720,
            height=576,
            duration_seconds=120.0,
            frame_rate_fps=25.0,
        ),
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
vast:
  gpu_models: [RTX 3090 Ti]
  max_hourly_price_usd: 0.35
  min_reliability: 0.99
  verified_only: true
  disk_gb: 50
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "estimate",
            str(video_path),
            "--fixture",
            "tests/fixtures/offers.json",
            "--fps-processing",
            "5",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "Cost Estimation" in result.stdout
    assert "simulation" in result.stdout.lower()
