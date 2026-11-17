import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from worker.process import (
    ProcessingReport,
    WorkerError,
    _build_staged_output_path,
    _fail,
    compute_target_canvas,
    main,
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


def test_build_staged_output_path_preserves_container_suffix() -> None:
    assert _build_staged_output_path(Path("/workspace/output/test-upscaled.mp4")).name == "test-upscaled.tmp.mp4"
    assert _build_staged_output_path(Path("/workspace/output/test-upscaled.mkv")).name == "test-upscaled.tmp.mkv"
    assert _build_staged_output_path(Path("/workspace/output/test-upscaled")).name == "test-upscaled.tmp"


def test_fail_report_preserves_frames_processed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("worker.process.time.perf_counter", lambda: 10.0)
    report_path = tmp_path / "report.json"
    args = Namespace(
        input=str(tmp_path / "in.mp4"),
        output=str(tmp_path / "out.mp4"),
        profile="balanced",
    )

    exit_code = _fail(
        report_path=report_path,
        args=args,
        started=8.0,
        notes=["encoding failed"],
        status="failed",
        frames_processed=7,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["frames_processed"] == 7
    assert payload["processing_fps"] > 0


def test_main_failure_after_processing_preserves_frames_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    report_path = tmp_path / "report.json"
    input_path.write_bytes(b"x")

    args = Namespace(
        input=str(input_path),
        output=str(output_path),
        report=str(report_path),
        target_width=1280,
        target_height=720,
        profile="balanced",
        video_crf=20,
        models_dir=str(tmp_path),
        min_free_gb=0.0,
    )
    monkeypatch.setattr("worker.process.parse_args", lambda: args)
    monkeypatch.setattr("worker.process._ensure_disk_space", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "worker.process.probe_video",
        lambda _path: SimpleNamespace(width=640, height=360, fps=24.0, duration_seconds=5.0),
    )
    monkeypatch.setattr("worker.process.compute_target_canvas", lambda **_k: (1280, 720))
    monkeypatch.setattr(
        "worker.process.select_model",
        lambda **_k: SimpleNamespace(
            name="realesrgan",
            weights_path=tmp_path / "weights.pth",
            model_name="RealESRGAN_x4plus",
            model_scale=4,
            tile=256,
            tile_pad=10,
            pre_pad=0,
        ),
    )

    class DummyUpscaler:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def load(self) -> None:
            return None

        @staticmethod
        def detect_gpu_name() -> str:
            return "NVIDIA GeForce RTX 3060"

    monkeypatch.setattr("worker.process.RealesrganUpscaler", DummyUpscaler)
    monkeypatch.setattr("worker.process.run_streaming_upscale", lambda **_k: 12)

    def _failing_mux(*, output_path: Path, **_kwargs) -> None:
        output_path.write_bytes(b"partial")
        raise WorkerError("mux failed")

    monkeypatch.setattr("worker.process.mux_with_source_streams", _failing_mux)

    exit_code = main()

    staged = output_path.with_name("output.tmp.mp4")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["frames_processed"] == 12
    assert payload["processing_fps"] > 0
    assert not staged.exists()
