import json
import sys
from argparse import Namespace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from worker.process import (
    PROGRESS_EVENT_PREFIX,
    ProcessingReport,
    ProgressEmitter,
    WorkerError,
    _build_staged_output_path,
    _emit_cuda_diagnostics,
    _estimate_total_frames,
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


def test_progress_emitter_outputs_json_event(monkeypatch: pytest.MonkeyPatch) -> None:
    out = StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    emitter = ProgressEmitter(started_at=0.0, emit_interval_seconds=10.0, emit_frame_interval=1000)
    monkeypatch.setattr("worker.process.time.perf_counter", lambda: 2.0)

    emitter.emit(stage="upscale", frames_done=20, frames_total=100, force=True)

    rendered = out.getvalue().strip()
    assert rendered.startswith(PROGRESS_EVENT_PREFIX)
    payload = json.loads(rendered[len(PROGRESS_EVENT_PREFIX) :])
    assert payload["stage"] == "upscale"
    assert payload["frames_done"] == 20
    assert payload["frames_total"] == 100
    assert payload["fps_processing"] == 10.0
    assert payload["eta_seconds"] == 8.0


def test_estimate_total_frames_returns_none_for_unknown_duration() -> None:
    assert _estimate_total_frames(duration_seconds=None, fps=24.0) is None
    assert _estimate_total_frames(duration_seconds=0.0, fps=24.0) is None
    assert _estimate_total_frames(duration_seconds=10.0, fps=0.0) is None
    assert _estimate_total_frames(duration_seconds=5.0, fps=24.0) == 120


def test_emit_cuda_diagnostics_logs_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA RTX 5090"

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (12, 0)

        @staticmethod
        def get_arch_list() -> list[str]:
            return ["sm_86", "sm_89", "sm_120"]

    class _Torch:
        __version__ = "2.14.1+cu130"
        cuda = _Cuda()
        version = SimpleNamespace(cuda="13.0")

    monkeypatch.setitem(sys.modules, "torch", _Torch())

    _emit_cuda_diagnostics()

    logged = stderr.getvalue()
    assert "torch=2.14.1+cu130" in logged
    assert "compiled_cuda=13.0" in logged
    assert "gpu=NVIDIA RTX 5090" in logged
    assert "capability=12.0" in logged
    assert "sm_120" in logged


def test_emit_cuda_diagnostics_rejects_unsupported_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA RTX 5090"

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (12, 0)

        @staticmethod
        def get_arch_list() -> list[str]:
            return ["sm_86", "sm_89"]

    class _Torch:
        __version__ = "2.14.1+cu130"
        cuda = _Cuda()
        version = SimpleNamespace(cuda="13.0")

    monkeypatch.setitem(sys.modules, "torch", _Torch())

    with pytest.raises(WorkerError) as exc:
        _emit_cuda_diagnostics()
    assert "incompatible" in str(exc.value).lower()
    assert "sm_120" in str(exc.value)
