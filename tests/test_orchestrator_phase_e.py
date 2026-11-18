from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

import vast_upscaler.orchestrator as orchestrator_module
from vast_upscaler.config import Settings
from vast_upscaler.instances import InstanceLifecycleError
from vast_upscaler.orchestrator import (
    CleanupItemResult,
    OrchestrationError,
    OrchestrationOptions,
    SessionExecutionLock,
    SessionLockError,
    SshEndpointProbeError,
    SshTrustValidationError,
    cleanup_sessions,
    ensure_store_readable,
    resume_session,
    run_upscale_workflow,
)
from vast_upscaler.provisioning_history import (
    ProvisioningHistoryCorruptionError,
)
from vast_upscaler.session_store import SessionRecord, SessionStore
from vast_upscaler.transfer import KnownHostKey, SshConnectionError
from vast_upscaler.vast_write_client import InstanceSummary, VastCreateAmbiguousError
from vast_upscaler.video import VideoInfo


def _fingerprint(key_data: str) -> str:
    digest = hashlib.sha256(base64.b64decode(key_data.encode("ascii"), validate=True)).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


@pytest.fixture(autouse=True)
def _stub_ssh_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    key_data = "QUFBQQ=="
    fp = _fingerprint(key_data)

    def _fake_fetch(**kwargs):
        host = kwargs["host"]
        port = kwargs["port"]
        return KnownHostKey(
            marker=f"[{host}]:{port}",
            key_type="ssh-ed25519",
            key_data=key_data,
            sha256_fingerprint=fp,
            source_line=f"[{host}]:{port} ssh-ed25519 {key_data}",
        )

    monkeypatch.setattr("vast_upscaler.orchestrator.fetch_ssh_ed25519_host_key", _fake_fetch)
    monkeypatch.setattr("vast_upscaler.orchestrator.validate_local_ssh_prerequisites", lambda **_k: None)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    monkeypatch.setattr("vast_upscaler.orchestrator.socket.create_connection", lambda *_a, **_k: _Conn())


class FakeReadClient:
    def __init__(self, offers: list[dict] | None = None) -> None:
        self.calls = 0
        self.offers = offers or [
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

    def search_offers(self, _query):
        self.calls += 1
        return list(self.offers)


class FakeWriteClient:
    def __init__(self) -> None:
        self.created = []
        self.destroyed = []
        self.create_raises = None
        self.destroy_ok = True
        self.list_instances_rows = []
        self.statuses = ["running"]
        self.get_calls = 0

    def create_instance(self, req):
        if self.create_raises is not None:
            raise self.create_raises
        self.created.append(req)
        return 123

    def get_instance(self, instance_id: int):
        status = self.statuses[min(self.get_calls, len(self.statuses) - 1)]
        self.get_calls += 1
        return InstanceSummary(
            instance_id=instance_id,
            actual_status=status,
            intended_status="running",
            cur_state="running" if status == "running" else "starting",
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

    def compare_create_payload_with_ssh_template(self, req):
        _ = req
        return {}


class FakeTransport:
    def __init__(self) -> None:
        self.ssh_ready_error = None
        self.ssh_ready_calls = 0

    def check_ssh_ready(self, _conn, *, timeout_seconds: float):
        _ = timeout_seconds
        self.ssh_ready_calls += 1
        if self.ssh_ready_error is not None:
            raise self.ssh_ready_error


class _MemoryHistoryStore:
    def __init__(self) -> None:
        self.attempts = []

    def append_attempt(self, attempt) -> None:
        self.attempts.append(attempt)


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
    known_hosts.write_text("[1.2.3.4]:2200 ssh-ed25519 QUFBQQ==\n", encoding="utf-8")
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


def test_provisioning_progress_reports_updates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    read = FakeReadClient()
    write = FakeWriteClient()
    write.statuses = ["loading", "running"]
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

    progress_events: list[dict[str, str]] = []
    run_upscale_workflow(
        settings=settings,
        options=options,
        session_store=store,
        read_client=read,
        write_client=write,
        transport=transport,
        confirm_paid=lambda _m: True,
        confirm_ssh_trust=lambda _m: True,
        report_provisioning_status=lambda payload: progress_events.append(payload),
    )

    statuses = [item["status"] for item in progress_events if item["phase"] in {"first", "update"}]
    assert "loading" in statuses
    assert "running" in statuses


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


def test_ssh_prerequisite_check_runs_before_paid_creation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr(
        "vast_upscaler.orchestrator.validate_local_ssh_prerequisites",
        lambda **_k: (_ for _ in ()).throw(OrchestrationError("local ssh precheck failed")),
    )

    with pytest.raises(OrchestrationError):
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


def test_provisioning_timeout_triggers_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()
    write.statuses = ["loading", "loading", "loading"]

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.time.sleep", lambda _s: None)

    values = iter([0.0, 0.2, 0.4, 10.5, 10.6])
    monkeypatch.setattr("vast_upscaler.instances.time.perf_counter", lambda: next(values))

    with pytest.raises(InstanceLifecycleError):
        run_upscale_workflow(
            settings=settings,
            options=OrchestrationOptions(**{**options.__dict__, "max_wait_ready_seconds": 10.0, "poll_interval_seconds": 0.0}),
            session_store=store,
            read_client=FakeReadClient(),
            write_client=write,
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )
    assert write.destroyed == [123]


def test_provisioning_timeout_cleanup_failure_marks_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()
    write.statuses = ["loading", "loading"]
    write.destroy_ok = False

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.time.sleep", lambda _s: None)

    values = iter([0.0, 0.2, 10.5])
    monkeypatch.setattr("vast_upscaler.instances.time.perf_counter", lambda: next(values))

    with pytest.raises(InstanceLifecycleError):
        run_upscale_workflow(
            settings=settings,
            options=OrchestrationOptions(**{**options.__dict__, "max_wait_ready_seconds": 10.0, "poll_interval_seconds": 0.0}),
            session_store=store,
            read_client=FakeReadClient(),
            write_client=write,
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )

    saved = store.current()
    assert saved is not None
    assert saved.state == "CLEANUP_REQUIRED"
    assert saved.cleanup_error == "Destroy call returned unsuccessful response"


def test_ssh_tcp_probe_immediate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="ssh9.vast.ai",
        ssh_port=17022,
        gpu_name="RTX 3090",
    )
    tracker = orchestrator_module.ProvisioningAttemptTracker(session_id="sess-1")
    calls = {"n": 0}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    def _fake_connect(*_args, **_kwargs):
        calls["n"] += 1
        return _Conn()

    monkeypatch.setattr("vast_upscaler.orchestrator.socket.create_connection", _fake_connect)
    events: list[dict[str, str]] = []

    orchestrator_module._wait_for_ssh_tcp_availability(
        instance=summary,
        timeout_seconds=75.0,
        retry_delay_seconds=5.0,
        report_provisioning_status=lambda payload: events.append(payload),
        tracker=tracker,
    )

    assert calls["n"] == 1
    assert any(item["status"] == "reachable" for item in events)


def test_ssh_tcp_probe_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="ssh9.vast.ai",
        ssh_port=17022,
        gpu_name="RTX 3090",
    )
    tracker = orchestrator_module.ProvisioningAttemptTracker(session_id="sess-1")
    calls = {"n": 0}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    def _fake_connect(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("Connection refused")
        return _Conn()

    monkeypatch.setattr("vast_upscaler.orchestrator.socket.create_connection", _fake_connect)
    monkeypatch.setattr("vast_upscaler.orchestrator.time.sleep", lambda _s: None)

    now_values = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr("vast_upscaler.orchestrator.time.monotonic", lambda: next(now_values))
    events: list[dict[str, str]] = []

    orchestrator_module._wait_for_ssh_tcp_availability(
        instance=summary,
        timeout_seconds=75.0,
        retry_delay_seconds=5.0,
        report_provisioning_status=lambda payload: events.append(payload),
        tracker=tracker,
    )

    assert calls["n"] == 3
    assert sum(1 for item in events if item["status"] == "retrying") == 2
    assert any(item["status"] == "reachable" for item in events)


def test_ssh_tcp_probe_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="ssh9.vast.ai",
        ssh_port=17022,
        gpu_name="RTX 3090",
    )
    tracker = orchestrator_module.ProvisioningAttemptTracker(session_id="sess-1")

    monkeypatch.setattr("vast_upscaler.orchestrator.socket.create_connection", lambda *_a, **_k: (_ for _ in ()).throw(OSError("Connection refused")))
    monkeypatch.setattr("vast_upscaler.orchestrator.time.sleep", lambda _s: None)
    now_values = iter([0.0, 40.0, 81.0, 82.0])
    monkeypatch.setattr("vast_upscaler.orchestrator.time.monotonic", lambda: next(now_values))

    with pytest.raises(SshEndpointProbeError):
        orchestrator_module._wait_for_ssh_tcp_availability(
            instance=summary,
            timeout_seconds=75.0,
            retry_delay_seconds=5.0,
            report_provisioning_status=lambda _payload: None,
            tracker=tracker,
        )


def test_ssh_tcp_probe_rejects_missing_or_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = orchestrator_module.ProvisioningAttemptTracker(session_id="sess-1")
    summary_missing = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="ssh9.vast.ai",
        ssh_port=None,
        gpu_name="RTX 3090",
    )
    with pytest.raises(SshEndpointProbeError):
        orchestrator_module._wait_for_ssh_tcp_availability(
            instance=summary_missing,
            timeout_seconds=75.0,
            retry_delay_seconds=5.0,
            report_provisioning_status=None,
            tracker=tracker,
        )

    summary_invalid = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="ssh9.vast.ai",
        ssh_port=70000,
        gpu_name="RTX 3090",
    )
    with pytest.raises(SshEndpointProbeError):
        orchestrator_module._wait_for_ssh_tcp_availability(
            instance=summary_invalid,
            timeout_seconds=75.0,
            retry_delay_seconds=5.0,
            report_provisioning_status=None,
            tracker=tracker,
        )


def test_ssh_endpoint_probe_timeout_triggers_cleanup_and_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()
    write.list_instances_rows = []
    history_store = _MemoryHistoryStore()

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr(
        "vast_upscaler.orchestrator._wait_for_ssh_tcp_availability",
        lambda **_k: (_ for _ in ()).throw(SshEndpointProbeError("refused")),
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.ProvisioningHistoryStore", lambda: history_store)

    with pytest.raises(SshEndpointProbeError):
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

    assert write.destroyed == [123]
    assert len(history_store.attempts) == 1
    assert history_store.attempts[0].outcome == "SSH_CONNECT_FAIL"


def test_interruption_during_ssh_endpoint_wait_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr(
        "vast_upscaler.orchestrator._wait_for_ssh_tcp_availability",
        lambda **_k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(OrchestrationError):
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

    saved = store.current()
    assert saved is not None
    assert saved.state in {"PRESERVED_FOR_RECOVERY", "DESTROYED"}


def test_ssh_bootstrap_failure_not_masked_by_endpoint_wait(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    unknown_known_hosts = tmp_path / "known_hosts_unknown"
    unknown_known_hosts.write_text("", encoding="utf-8")
    options = OrchestrationOptions(**{**options.__dict__, "known_hosts_path": unknown_known_hosts})
    store = SessionStore(path=tmp_path / "sessions.json")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    calls = {"n": 0}

    def _probe_ok(**_k):
        calls["n"] += 1

    monkeypatch.setattr("vast_upscaler.orchestrator._wait_for_ssh_tcp_availability", _probe_ok)
    monkeypatch.setattr(
        "vast_upscaler.orchestrator.execute_remote_worker",
        lambda **_k: (_ for _ in ()).throw(AssertionError("worker should not run")),
    )

    with pytest.raises(SshTrustValidationError):
        run_upscale_workflow(
            settings=settings,
            options=OrchestrationOptions(**{**options.__dict__, "ssh_bootstrap_mode": "tofu"}),
            session_store=store,
            read_client=FakeReadClient(),
            write_client=FakeWriteClient(),
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: False,
        )

    assert calls["n"] == 1


def test_provisioning_history_records_success_and_deduplicates_status_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()
    write.statuses = ["loading", "loading", "running"]
    write.list_instances_rows = []
    history_store = _MemoryHistoryStore()

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.ProvisioningHistoryStore", lambda: history_store)

    class _WorkerResult:
        local_output_path = options.output_path
        local_report_path = options.report_path
        frames_processed = 100
        video_info = VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0)

    monkeypatch.setattr("vast_upscaler.orchestrator.execute_remote_worker", lambda **_k: _WorkerResult())

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

    assert summary.metrics_warning is None
    assert len(history_store.attempts) == 1
    attempt = history_store.attempts[0]
    assert attempt.outcome == "PROVISION_SUCCESS"
    assert attempt.provisioning_succeeded is True
    assert attempt.processing_succeeded is True
    status_events = [e for e in attempt.events if e.kind == "status"]
    assert len(status_events) == 2
    assert status_events[0].details["status"] == "loading"
    assert status_events[1].details["status"] == "running"


def test_provisioning_history_records_timeout_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()
    write.statuses = ["loading", "loading", "loading"]
    history_store = _MemoryHistoryStore()

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.time.sleep", lambda _s: None)
    monkeypatch.setattr("vast_upscaler.orchestrator.ProvisioningHistoryStore", lambda: history_store)

    values = iter([0.0, 0.2, 0.4, 10.5, 10.6])
    monkeypatch.setattr("vast_upscaler.instances.time.perf_counter", lambda: next(values))

    with pytest.raises(InstanceLifecycleError):
        run_upscale_workflow(
            settings=settings,
            options=OrchestrationOptions(**{**options.__dict__, "max_wait_ready_seconds": 10.0, "poll_interval_seconds": 0.0}),
            session_store=store,
            read_client=FakeReadClient(),
            write_client=write,
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )

    assert len(history_store.attempts) == 1
    attempt = history_store.attempts[0]
    assert attempt.outcome == "PROVISION_TIMEOUT"
    assert attempt.provisioning_succeeded is False
    assert attempt.processing_succeeded is False


def test_provisioning_history_write_warning_does_not_fail_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = _options(tmp_path)
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()
    write.list_instances_rows = []

    class _BrokenHistoryStore:
        def append_attempt(self, _attempt) -> None:
            raise ProvisioningHistoryCorruptionError("broken file")

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.ProvisioningHistoryStore", lambda: _BrokenHistoryStore())

    class _WorkerResult:
        local_output_path = options.output_path
        local_report_path = options.report_path
        frames_processed = 100
        video_info = VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0)

    monkeypatch.setattr("vast_upscaler.orchestrator.execute_remote_worker", lambda **_k: _WorkerResult())
    warnings: list[dict[str, str]] = []

    summary = run_upscale_workflow(
        settings=settings,
        options=options,
        session_store=store,
        read_client=FakeReadClient(),
        write_client=write,
        transport=FakeTransport(),
        confirm_paid=lambda _m: True,
        confirm_ssh_trust=lambda _m: True,
        report_provisioning_status=lambda payload: warnings.append(payload),
    )

    assert summary.state == "DESTROYED"
    assert summary.metrics_warning is not None
    assert "corrupted" in summary.metrics_warning
    assert any(item["phase"] == "metrics-warning" for item in warnings)


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


def test_explicit_offer_id_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = OrchestrationOptions(**{**_options(tmp_path).__dict__, "dry_run": True, "offer_id": 40002})
    store = SessionStore(path=tmp_path / "sessions.json")
    read = FakeReadClient(
        offers=[
            {
                "id": 40001,
                "gpu_name": "RTX 3090",
                "dph_total": 0.21,
                "reliability": 0.999,
                "disk_space": 120,
                "num_gpus": 1,
                "direct_port_count": 1,
                "verified": True,
                "search": {"totalHour": 0.21, "gpuCostPerHour": 0.2, "diskHour": 0.01},
            },
            {
                "id": 40002,
                "gpu_name": "RTX 3090",
                "dph_total": 0.25,
                "reliability": 0.999,
                "disk_space": 120,
                "num_gpus": 1,
                "direct_port_count": 1,
                "verified": True,
                "search": {"totalHour": 0.25, "gpuCostPerHour": 0.23, "diskHour": 0.02},
            },
        ]
    )

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    summary = run_upscale_workflow(
        settings=settings,
        options=options,
        session_store=store,
        read_client=read,
        write_client=FakeWriteClient(),
        transport=FakeTransport(),
        confirm_paid=lambda _m: True,
        confirm_ssh_trust=lambda _m: True,
    )
    assert summary.offer_id == 40002


def test_explicit_offer_id_not_found_no_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = OrchestrationOptions(**{**_options(tmp_path).__dict__, "offer_id": 99999})
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    with pytest.raises(OrchestrationError):
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

    assert write.created == []


def test_explicit_offer_id_incompatible_no_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = OrchestrationOptions(**{**_options(tmp_path).__dict__, "offer_id": 77777})
    store = SessionStore(path=tmp_path / "sessions.json")
    write = FakeWriteClient()
    read = FakeReadClient(
        offers=[
            {
                "id": 77777,
                "gpu_name": "RTX 3090",
                "dph_total": 0.8,
                "reliability": 0.999,
                "disk_space": 120,
                "num_gpus": 1,
                "direct_port_count": 1,
                "verified": True,
                "search": {"totalHour": 0.8, "gpuCostPerHour": 0.79, "diskHour": 0.01},
            }
        ]
    )

    monkeypatch.setattr(
        "vast_upscaler.orchestrator.inspect_video",
        lambda _p: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=5.0, frame_rate_fps=24.0),
    )

    with pytest.raises(OrchestrationError):
        run_upscale_workflow(
            settings=settings,
            options=options,
            session_store=store,
            read_client=read,
            write_client=write,
            transport=FakeTransport(),
            confirm_paid=lambda _m: True,
            confirm_ssh_trust=lambda _m: True,
        )

    assert write.created == []


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


def test_ssh_bootstrap_unknown_key_non_standard_port_adds_known_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = tmp_path / "id"
    key.write_text("x", encoding="utf-8")
    kh = tmp_path / "known_hosts"
    kh.write_text("", encoding="utf-8")
    instance = InstanceSummary(
        instance_id=1,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="1.2.3.4",
        ssh_port=17022,
    )
    scanned = KnownHostKey(
        marker="[1.2.3.4]:17022",
        key_type="ssh-ed25519",
        key_data="QUFBQQ==",
        sha256_fingerprint="SHA256:yzxHzFhR4QCG3SqLr3DB0uV8nY5M5D7G6g0j2A9M7s0",
        source_line="[1.2.3.4]:17022 ssh-ed25519 QUFBQQ==",
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.fetch_ssh_ed25519_host_key", lambda **_k: scanned)

    conn = orchestrator_module._build_trusted_connection(
        instance=instance,
        private_key_path=key,
        known_hosts_path=kh,
        confirm_ssh_trust=lambda _m: True,
        bootstrap_mode="tofu",
        request_verified_fingerprint=None,
    )
    assert conn.port == 17022
    assert scanned.source_line in kh.read_text(encoding="utf-8")


def test_ssh_bootstrap_rejects_unknown_key_when_tofu_not_confirmed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = tmp_path / "id"
    key.write_text("x", encoding="utf-8")
    kh = tmp_path / "known_hosts"
    kh.write_text("", encoding="utf-8")
    instance = InstanceSummary(
        instance_id=1,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="1.2.3.4",
        ssh_port=17022,
    )
    scanned = KnownHostKey(
        marker="[1.2.3.4]:17022",
        key_type="ssh-ed25519",
        key_data="QUFBQQ==",
        sha256_fingerprint="SHA256:expected",
        source_line="[1.2.3.4]:17022 ssh-ed25519 QUFBQQ==",
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.fetch_ssh_ed25519_host_key", lambda **_k: scanned)

    with pytest.raises(SshTrustValidationError):
        orchestrator_module._build_trusted_connection(
            instance=instance,
            private_key_path=key,
            known_hosts_path=kh,
            confirm_ssh_trust=lambda _m: False,
            bootstrap_mode="tofu",
            request_verified_fingerprint=None,
        )


def test_ssh_bootstrap_rejects_wrong_verified_fingerprint_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = tmp_path / "id"
    key.write_text("x", encoding="utf-8")
    kh = tmp_path / "known_hosts"
    kh.write_text("", encoding="utf-8")
    instance = InstanceSummary(
        instance_id=1,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="1.2.3.4",
        ssh_port=17022,
    )
    scanned = KnownHostKey(
        marker="[1.2.3.4]:17022",
        key_type="ssh-ed25519",
        key_data="QUFBQQ==",
        sha256_fingerprint="SHA256:expected",
        source_line="[1.2.3.4]:17022 ssh-ed25519 QUFBQQ==",
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.fetch_ssh_ed25519_host_key", lambda **_k: scanned)

    with pytest.raises(SshTrustValidationError):
        orchestrator_module._build_trusted_connection(
            instance=instance,
            private_key_path=key,
            known_hosts_path=kh,
            confirm_ssh_trust=lambda _m: True,
            bootstrap_mode="strict",
            request_verified_fingerprint=lambda _m: "SHA256:other",
        )


def test_ssh_bootstrap_rejects_changed_approved_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = tmp_path / "id"
    key.write_text("x", encoding="utf-8")
    kh = tmp_path / "known_hosts"
    kh.write_text("[1.2.3.4]:17022 ssh-ed25519 QUFBQQ==\n", encoding="utf-8")
    instance = InstanceSummary(
        instance_id=1,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="1.2.3.4",
        ssh_port=17022,
    )
    scanned = KnownHostKey(
        marker="[1.2.3.4]:17022",
        key_type="ssh-ed25519",
        key_data="QkJCQg==",
        sha256_fingerprint="SHA256:new",
        source_line="[1.2.3.4]:17022 ssh-ed25519 QkJCQg==",
    )
    monkeypatch.setattr("vast_upscaler.orchestrator.fetch_ssh_ed25519_host_key", lambda **_k: scanned)

    with pytest.raises(SshTrustValidationError):
        orchestrator_module._build_trusted_connection(
            instance=instance,
            private_key_path=key,
            known_hosts_path=kh,
            confirm_ssh_trust=lambda _m: True,
            bootstrap_mode="tofu",
            request_verified_fingerprint=lambda _m: scanned.sha256_fingerprint,
        )
