import json
from pathlib import Path

import pytest

from worker.process import (
    ProcessingReport,
    WorkerError,
    compute_target_canvas,
    validate_output,
    write_report,
)


def test_compute_target_canvas() -> None:
    tw, th = compute_target_canvas(720, 576, 1920, 1080)
    assert tw == 1920
    assert th == 1080


def test_compute_target_canvas_invalid() -> None:
    with pytest.raises(WorkerError):
        compute_target_canvas(0, 576, 1920, 1080)


def test_write_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = ProcessingReport(
        status="completed",
        source="in.mkv",
        output="out.mkv",
        model="realesrgan",
        profile="balanced",
        frames_processed=100,
        processing_seconds=50.0,
        processing_fps=2.0,
        gpu="RTX 3090",
        notes=["ok"],
    )

    write_report(report_path, report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["frames_processed"] == 100


def test_validate_output_fps_tolerance(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "out.mkv"
    output_path.write_bytes(b"x")

    class DummyProbe:
        width = 1920
        height = 1080
        fps = 25.05
        duration_seconds = 100.2

    monkeypatch.setattr("worker.process.probe_video", lambda _path: DummyProbe())

    validate_output(output_path=output_path, source_duration=100.0, source_fps=25.0)
