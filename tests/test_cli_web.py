from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from vast_upscaler.cli import app

runner = CliRunner()


def test_web_command_binds_localhost_only(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["web", "--config", str(cfg), "--host", "0.0.0.0"])
    assert result.exit_code == 2
    assert "127.0.0.1 only" in result.output


def test_web_command_starts_uvicorn(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run(app_instance, host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["app"] = app_instance

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=_fake_run))

    result = runner.invoke(app, ["web", "--config", str(cfg), "--host", "127.0.0.1", "--port", "8765"])
    assert result.exit_code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
