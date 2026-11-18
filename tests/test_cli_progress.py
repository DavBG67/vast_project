from __future__ import annotations

from io import StringIO

import pytest

from vast_upscaler import cli


class _StdoutCapture(StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_worker_progress_reporter_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = _StdoutCapture(tty=True)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    lines: list[str] = []
    monkeypatch.setattr(cli.console, "print", lambda msg: lines.append(str(msg)))

    reporter = cli._WorkerProgressReporter()
    reporter({"kind": "worker", "stage": "upscale", "frames_done": 25, "frames_total": 100, "fps_processing": 12.3, "eta_seconds": 6.1})
    reporter({"kind": "transfer", "stage": "upload", "status": "completed"})
    reporter.close()

    assert any("Worker stage" in line for line in lines)
    assert any("Transfer upload" in line for line in lines)
    assert "Upscale 25.0% | 25/100 frames | 12.3 fps | ETA 6s" in stdout.getvalue()


def test_worker_progress_reporter_redirected_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = _StdoutCapture(tty=False)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    lines: list[str] = []
    monkeypatch.setattr(cli.console, "print", lambda msg: lines.append(str(msg)))
    ticks = iter([0.0, 0.0, 6.5])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(ticks))

    reporter = cli._WorkerProgressReporter()
    reporter({"kind": "worker", "stage": "decoding", "frames_done": 0})
    reporter({"kind": "worker", "stage": "decoding", "frames_done": 1})
    reporter({"kind": "worker", "stage": "decoding", "frames_done": 2})

    non_stage_lines = [line for line in lines if line.startswith("Upscale ")]
    assert len(non_stage_lines) == 2
