from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from vast_upscaler.cli import app
from vast_upscaler.orchestrator import OrchestrationSummary
from vast_upscaler.video import VideoInfo

runner = CliRunner()


class _FakeReadClient:
    def close(self) -> None:
        return


class _FakeWriteClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return None


def _config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
secrets:
  vast_api_key: test
  ssh_private_key_path: C:/tmp/id
video:
  default_profile: balanced
  target_width: 1280
  target_height: 720
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return cfg


def _summary() -> OrchestrationSummary:
    return OrchestrationSummary(
        session_id="sess-demo",
        state="COMPLETED",
        offer_id=31129,
        offer_gpu="RTX 3090",
        estimate=SimpleNamespace(total_estimated_usd=0.12, rental_total_hourly_usd=0.2),
        video_input=VideoInfo(file_path="in.mp4", width=1280, height=720, duration_seconds=5.0),
        worker_result=SimpleNamespace(
            local_output_path=Path("out.mp4"),
            local_report_path=Path("out.mp4.report.json"),
            frames_processed=120,
            video_info=VideoInfo(file_path="out.mp4", width=1280, height=720, duration_seconds=5.0),
        ),
        instance_id=123,
        cleanup_verified=True,
        cleanup_warning=None,
    )


def test_upscale_dry_run_cli_with_simulated_deps(monkeypatch, tmp_path: Path) -> None:
    cfg = _config_file(tmp_path)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")

    monkeypatch.setattr("vast_upscaler.cli._build_read_client", lambda settings: _FakeReadClient())
    monkeypatch.setattr("vast_upscaler.cli._build_write_client", lambda settings: _FakeWriteClient())
    monkeypatch.setattr("vast_upscaler.cli.SessionExecutionLock", _NoLock)
    monkeypatch.setattr("vast_upscaler.cli.ensure_store_readable", lambda store: None)

    captured = {}

    def _fake_run(**kwargs):
        captured["dry_run"] = kwargs["options"].dry_run
        captured["offer_id"] = kwargs["options"].offer_id
        captured["max_wait_ready_seconds"] = kwargs["options"].max_wait_ready_seconds
        return _summary()

    monkeypatch.setattr("vast_upscaler.cli.run_upscale_workflow", _fake_run)

    result = runner.invoke(
        app,
        [
            "upscale",
            str(src),
            "--config",
            str(cfg),
            "--worker-image",
            "ghcr.io/acme/worker:test",
            "--dry-run",
            "--offer-id",
            "31129",
            "--provisioning-timeout-seconds",
            "650",
        ],
    )

    assert result.exit_code == 0
    assert captured["dry_run"] is True
    assert captured["offer_id"] == 31129
    assert captured["max_wait_ready_seconds"] == 650
    assert "Upscale Summary (DRY-RUN)" in result.stdout


def test_upscale_cli_handles_critical_failure(monkeypatch, tmp_path: Path) -> None:
    cfg = _config_file(tmp_path)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")

    monkeypatch.setattr("vast_upscaler.cli._build_read_client", lambda settings: _FakeReadClient())
    monkeypatch.setattr("vast_upscaler.cli._build_write_client", lambda settings: _FakeWriteClient())
    monkeypatch.setattr("vast_upscaler.cli.SessionExecutionLock", _NoLock)
    monkeypatch.setattr("vast_upscaler.cli.ensure_store_readable", lambda store: None)
    monkeypatch.setattr(
        "vast_upscaler.cli.run_upscale_workflow",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated fatal")),
    )

    result = runner.invoke(
        app,
        [
            "upscale",
            str(src),
            "--config",
            str(cfg),
            "--worker-image",
            "ghcr.io/acme/worker:test",
        ],
    )

    assert result.exit_code == 1
    assert "Upscale failed: simulated fatal" in result.stdout


def test_instances_destroy_resume_cleanup_cli_paths(monkeypatch, tmp_path: Path) -> None:
    cfg = _config_file(tmp_path)

    monkeypatch.setattr("vast_upscaler.cli._build_write_client", lambda settings: _FakeWriteClient())
    monkeypatch.setattr("vast_upscaler.cli.list_active_instances", lambda client: [{"id": 1, "label": "a", "actual_status": "running", "cur_state": "running"}])
    monkeypatch.setattr("vast_upscaler.cli.destroy_instance_with_verification", lambda **kwargs: True)
    monkeypatch.setattr(
        "vast_upscaler.cli.resume_session",
        lambda **kwargs: SimpleNamespace(session_id="sess-1", state="PRESERVED_FOR_RECOVERY", message="manual"),
    )
    monkeypatch.setattr(
        "vast_upscaler.cli.cleanup_sessions",
        lambda **kwargs: [SimpleNamespace(session_id="sess-1", attempted=True, cleaned=True, warning=None)],
    )
    monkeypatch.setattr("vast_upscaler.cli.SessionExecutionLock", _NoLock)

    r1 = runner.invoke(app, ["instances", "--config", str(cfg)])
    r2 = runner.invoke(app, ["destroy", "1", "--config", str(cfg), "--yes"])
    r3 = runner.invoke(app, ["session", "resume", "--config", str(cfg)])
    r4 = runner.invoke(app, ["session", "cleanup", "--config", str(cfg)])

    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert r3.exit_code == 0
    assert r4.exit_code == 0
    assert "Active Vast.ai Instances" in r1.stdout
    assert "destroyed and disappearance verified" in r2.stdout
    assert "Session sess-1" in r3.stdout
    assert "Session Cleanup" in r4.stdout


def test_instances_reports_readable_error_when_api_key_missing(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("video:\n  default_profile: balanced\n", encoding="utf-8")

    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VAST_SSH_PRIVATE_KEY", raising=False)

    result = runner.invoke(app, ["instances", "--config", str(cfg)])

    assert result.exit_code == 1
    assert "Failed to list instances: VAST_API_KEY missing." in result.stdout
    assert "Traceback" not in result.stdout


def test_confirm_ssh_trust_refuses_non_interactive(monkeypatch) -> None:
    class _FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return False

    monkeypatch.setattr("vast_upscaler.cli.sys.stdin", _FakeStdin())
    from vast_upscaler.cli import _confirm_ssh_trust

    assert _confirm_ssh_trust("msg") is False
