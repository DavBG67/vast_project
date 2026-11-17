from __future__ import annotations

from pathlib import Path

import pytest

from vast_upscaler.config import Settings
from vast_upscaler.orchestrator import (
    CleanupItemResult,
    OrchestrationError,
    OrchestrationOptions,
    SessionExecutionLock,
    SessionLockError,
    cleanup_sessions,
    ensure_store_readable,
    resume_session,
    run_upscale_workflow,
)
from vast_upscaler.session_store import SessionRecord, SessionStore
from vast_upscaler.transfer import SshConnectionError
from vast_upscaler.vast_write_client import InstanceSummary, VastCreateAmbiguousError
from vast_upscaler.video import VideoInfo


class FakeReadClient:
    def __init__(self) -> None:
        self.calls = 0

    def search_offers(self, _query):
        self.calls += 1
        return [
            {
                "id": 31129,
                "gpu_name": "RTX 3090",
                "dph_total": 0.22,
                "reliability": 0.999,
                "disk_space": 120,
                "num_gpus": 1,
                "direct_port_count": 1,
                "verified": True,
                "search": {"totalHour": 0.22, "gpuCostPerHour": 0.2, "diskHour": 0.02},
            }
        ]


class FakeWriteClient:
    def __init__(self) -> None:
        self.created = []
        self.destroyed = []
        self.create_raises = None
        self.destroy_ok = True
        self.list_instances_rows = []

    def create_instance(self, req):
        if self.create_raises is not None:
            raise self.create_raises
        self.created.append(req)
        return 123

    def get_instance(self, instance_id: int):
        return InstanceSummary(
            instance_id=instance_id,
            actual_status="running",
            intended_status="running",
            cur_state="running",
            label="sess",
            image_runtype="ssh_direct",
            ssh_host="1.2.3.4",
            ssh_port=2200,
        )

    def destroy_instance(self, instance_id: int):
        self.destroyed.append(instance_id)
        return self.destroy_ok

    def list_instances(self, *, limit=25, select_filters=None):
        _ = (limit, select_filters)
        return list(self.list_instances_rows)


class FakeTransport:
    def __init__(self) -> None:
        self.ssh_ready_error = None
        self.ssh_ready_calls = 0

    def check_ssh_ready(self, _conn, *, timeout_seconds: float):
        _ = timeout_seconds
        self.ssh_ready_calls += 1
        if self.ssh_ready_error is not None:
            raise self.ssh_ready_error


def _settings(tmp_path: Path) -> Settings:
    key = tmp_path / "id"
    key.write_text("x", encoding="utf-8")
    return Settings.model_validate(
        {
            "vast": {
                "gpu_models": ["RTX 3090"],
                "max_hourly_price_usd": 0.3,
                "min_reliability": 0.99,
                "verified_only": True,
                "disk_gb": 50,
                "max_session_hours": 12,
            },
            "video": {
                "target_width": 1920,
                "target_height": 1080,
                "default_profile": "balanced",
                "output_container": "mkv",
                "video_codec": "hevc",
            },
            "secrets": {
                "vast_api_key": "k",
                "ssh_private_key_path": str(key),
            },
        }
    )


def _options(tmp_path: Path) -> OrchestrationOptions:
    input_path = tmp_path / "in video.mp4"
    input_path.write_bytes(b"x")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("[1.2.3.4]:2200 ssh-ed25519 AAAA\n", encoding="utf-8")
    return OrchestrationOptions(
        input_path=input_path,
        output_path=tmp_path / "out" / "result.mp4",
        report_path=tmp_path / "out" / "report.json",
        worker_image="ghcr.io/acme/worker:tag",
        profile="balanced",
        target_width=1920,
        target_height=1080,
        known_hosts_path=known_hosts,
        processing_fps=5.0,
        poll_interval_seconds=0.0,
    )


def test_nominal_full_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    read = FakeReadClient()
    write = FakeWriteClient()
    write.list_instances_rows = []
    transport = FakeTransport()

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=30.0, frame_rate_fps=30.0),
    )

    class _WorkerResult:
        local_output_path = options.output_path
        local_report_path = options.report_path
        frames_processed = 900
        video_info = VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=30.0)

    monkeypatch.setattr("vast_upscaler.orchestrator.execute_remote_worker", lambda **_k: _WorkerResult())

    summary = run_upscale_workflow(
        settings=settings,
        options=options,
        session_store=store,
        read_client=read,
        write_client=write,
        transport=transport,
        confirm_paid=lambda _m: True,
        confirm_ssh_trust=lambda _m: True,
    )

    assert summary.state == "DESTROYED"
    assert summary.cleanup_verified is True
    assert write.created
    assert write.destroyed == [123]


def test_cancel_before_rental(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    write = FakeWriteClient()
    with pytest.raises(OrchestrationError):
        run_upscale_workflow(
            settings=settings,
            options=options,
            session_store=store,
            read_client=FakeReadClient(),
            write_client=write,
            transport=FakeTransport(),
            confirm_paid=lambda _m: False,
            confirm_ssh_trust=lambda _m: True,
        )
    assert write.created == []


def test_dry_run_no_paid_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    options = OrchestrationOptions(**{**options.__dict__, "dry_run": True})
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    write = FakeWriteClient()
    summary = run_upscale_workflow(
        settings=settings,
        options=options,
        session_store=store,
        read_client=FakeReadClient(),
        write_client=write,
        transport=FakeTransport(),
        confirm_paid=lambda _m: True,
        confirm_ssh_trust=lambda _m: True,
    )
    assert summary.instance_id is None
    assert write.created == []


def test_ambiguous_creation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    write = FakeWriteClient()
    write.create_raises = VastCreateAmbiguousError("ambiguous")

    with pytest.raises(VastCreateAmbiguousError):
        run_upscale_workflow(
            settings=settings,
            options=options,
            session_store=store,
            read_client=FakeReadClient(),
            write_client=write,
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )


def test_ssh_unavailable_triggers_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    write = FakeWriteClient()
    transport = FakeTransport()
    transport.ssh_ready_error = SshConnectionError("ssh down")

    with pytest.raises(SshConnectionError):
        run_upscale_workflow(
            settings=settings,
            options=options,
            session_store=store,
            read_client=FakeReadClient(),
            write_client=write,
            transport=transport,
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )
    assert write.destroyed == [123]


def test_worker_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.execute_remote_worker", lambda **_k: (_ for _ in ()).throw(RuntimeError("worker failed")))

    write = FakeWriteClient()
    with pytest.raises(RuntimeError):
        run_upscale_workflow(
            settings=settings,
            options=options,
            session_store=store,
            read_client=FakeReadClient(),
            write_client=write,
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )


def test_download_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vast_upscaler.remote_worker import RemoteWorkerOutputError

    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr(
        "vast_upscaler.orchestrator.execute_remote_worker",
        lambda **_k: (_ for _ in ()).throw(RemoteWorkerOutputError("download failed")),
    )

    with pytest.raises(RemoteWorkerOutputError):
        run_upscale_workflow(
            settings=settings,
            options=options,
            session_store=store,
            read_client=FakeReadClient(),
            write_client=FakeWriteClient(),
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )


def test_destroy_verification_failure_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    class _WorkerResult:
        local_output_path = options.output_path
        local_report_path = options.report_path
        frames_processed = 100
        video_info = VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0)

    monkeypatch.setattr("vast_upscaler.orchestrator.execute_remote_worker", lambda **_k: _WorkerResult())

    write = FakeWriteClient()
    write.list_instances_rows = [{"id": 123, "actual_status": "running"}]

    summary = run_upscale_workflow(
        settings=settings,
        options=options,
        session_store=store,
        read_client=FakeReadClient(),
        write_client=write,
        transport=FakeTransport(),
        confirm_paid=lambda _m: True,
        confirm_ssh_trust=lambda _m: True,
    )
    assert summary.cleanup_verified is False
    assert summary.cleanup_warning is not None


def test_local_interrupt_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr(
        "vast_upscaler.orchestrator.execute_remote_worker",
        lambda **_k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(OrchestrationError):
        run_upscale_workflow(
            settings=settings,
            options=options,
            session_store=store,
            read_client=FakeReadClient(),
            write_client=FakeWriteClient(),
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )


def test_resume_after_crash_recovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = SessionStore(path=tmp_path / "sessions.json")
    session = SessionRecord(session_id="s1", state="PRESERVED_FOR_RECOVERY", instance_id="123", input_path="x", output_path=str(tmp_path / "out.mkv"))
    session.mark_instance_owned("123", label="s1")
    store.append(session)

    settings = _settings(tmp_path)
    transport = FakeTransport()

    monkeypatch.setattr("vast_upscaler.orchestrator.recover_remote_results", lambda **_k: object())

    summary = resume_session(
        store=store,
        session_id="s1",
        write_client=FakeWriteClient(),
        settings=settings,
        transport=transport,
        known_hosts_path=_options(tmp_path).known_hosts_path,
        ssh_private_key_path=Path(settings.secrets.ssh_private_key_path),
        confirm_ssh_trust=lambda _m: True,
    )
    assert summary.state == "COMPLETED"


def test_cleanup_refuses_external_instance(tmp_path: Path) -> None:
    store = SessionStore(path=tmp_path / "sessions.json")
    session = SessionRecord(session_id="s-ext", state="FAILED", instance_id="999", managed_instance_id="111", instance_owned_by_session=False)
    store.append(session)

    results = cleanup_sessions(store=store, client=FakeWriteClient())
    assert len(results) == 1
    assert isinstance(results[0], CleanupItemResult)
    assert results[0].attempted is False
    assert results[0].warning is not None


def test_concurrent_execution_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock"
    with SessionExecutionLock(lock_path=lock_path), pytest.raises(SessionLockError), SessionExecutionLock(lock_path=lock_path):
        pass


def test_corrupted_session_store_refused(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text("{broken", encoding="utf-8")
    store = SessionStore(path=path)
    with pytest.raises(OrchestrationError):
        ensure_store_readable(store)
