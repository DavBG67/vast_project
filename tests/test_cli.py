from pathlib import Path

from typer.testing import CliRunner

from vast_upscaler.cli import app
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


def test_offers_offline_fixture() -> None:
    result = runner.invoke(app, ["offers", "--fixture", "tests/fixtures/offers.json"])

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

    result = runner.invoke(
        app,
        [
            "estimate",
            str(video_path),
            "--fixture",
            "tests/fixtures/offers.json",
            "--fps-processing",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "Cost Estimation" in result.stdout
    assert "simulation" in result.stdout.lower()
