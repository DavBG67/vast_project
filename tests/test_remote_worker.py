from __future__ import annotations

import json
from pathlib import Path

import pytest

from vast_upscaler.remote_worker import (
    RemoteWorkerAmbiguousStateError,
    RemoteWorkerError,
    RemoteWorkerOutputError,
    RemoteWorkerProcessError,
    RemoteWorkerReportError,
    RemoteWorkerRequest,
    RemoteWorkerTransferError,
    execute_remote_worker,
)
from vast_upscaler.transfer import (
    RemoteCommandResult,
    SshConnectionError,
    SshConnectionInfo,
    SshTimeoutError,
    TransferError,
)
from vast_upscaler.video import VideoInfo, VideoProbeError


class FakeTransport:
    def __init__(self) -> None:
        self.upload_error: Exception | None = None
        self.process_error: Exception | None = None
        self.probe_error: Exception | None = None
        self.process_result = RemoteCommandResult(returncode=0, stdout="worker ok\n", stderr="")
        self.probe_result = RemoteCommandResult(returncode=0, stdout="REPORT_PRESENT=1\nOUTPUT_PRESENT=1\n", stderr="")
        self.prepare_error: Exception | None = None
        self.report_payload: bytes = b"{}"
        self.output_payload: bytes = b"video-bytes"
        self.fail_report_download = False
        self.fail_output_download = False
        self.upload_calls: list[tuple[Path, str]] = []
        self.command_calls: list[list[str] | str] = []

    def upload_file(self, _conn, *, local_path, remote_dir, **_kwargs):
        if self.upload_error is not None:
            raise self.upload_error
        self.upload_calls.append((Path(local_path), str(remote_dir)))
        return f"{remote_dir}/{Path(local_path).name}"

    def run_remote_command(self, _conn, *, command, **_kwargs):
        self.command_calls.append(command)
        if isinstance(command, str) and command.startswith("mkdir -p "):
            if self.prepare_error is not None:
                raise self.prepare_error
            return RemoteCommandResult(returncode=0, stdout="", stderr="")
        if isinstance(command, str):
            if self.probe_error is not None:
                raise self.probe_error
            return self.probe_result
        if self.process_error is not None:
            raise self.process_error
        return self.process_result

    def download_file(self, _conn, *, remote_path, local_path, **_kwargs):
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if remote_path.endswith("/report.json"):
            if self.fail_report_download:
                raise TransferError("report missing")
            target.write_bytes(self.report_payload)
            return target
        if self.fail_output_download:
            raise TransferError("output missing")
        target.write_bytes(self.output_payload)
        return target


def _conn(tmp_path: Path) -> SshConnectionInfo:
    key = tmp_path / "id_ed25519"
    key.write_text("k", encoding="utf-8")
    kh = tmp_path / "known_hosts"
    kh.write_text("h", encoding="utf-8")
    return SshConnectionInfo(
        host="1.2.3.4",
        port=2222,
        username="root",
        private_key_path=key,
        known_hosts_path=kh,
        mode="proxy",
    )


def _request(tmp_path: Path) -> RemoteWorkerRequest:
    source_dir = tmp_path / "My Videos"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "input clip.mp4"
    source.write_bytes(b"src")
    return RemoteWorkerRequest(
        session_id="session D/01",
        local_input_path=source,
        local_output_path=tmp_path / "out" / "result video.mp4",
        local_report_path=tmp_path / "out" / "report.json",
        target_width=1920,
        target_height=1080,
        profile="balanced",
    )


def _completed_report(output_path: str) -> bytes:
    payload = {
        "status": "completed",
        "output": output_path,
        "frames_processed": 42,
    }
    return json.dumps(payload).encode("utf-8")


def test_execute_remote_worker_nominal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()

    remote_output = "/workspace/upscale-sessions/session-D-01/output/result video.mp4"
    transport.report_payload = _completed_report(remote_output)

    monkeypatch.setattr(
        "vast_upscaler.remote_worker.inspect_video",
        lambda *_a, **_k: VideoInfo(
            file_path=str(request.local_output_path),
            width=1920,
            height=1080,
            duration_seconds=12.0,
        ),
    )
    monkeypatch.setattr(
        "vast_upscaler.remote_worker.run_ffprobe",
        lambda *_a, **_k: {"streams": [{"codec_type": "video", "nb_frames": "42"}]},
    )

    result = execute_remote_worker(transport=transport, connection=conn, request=request)
    assert result.frames_processed == 42
    assert result.process_returncode == 0
    assert result.video_info.width == 1920
    assert result.output_frame_count == 42
    assert transport.upload_calls
    assert str(request.local_input_path) == str(transport.upload_calls[0][0])
    prepare_cmd = next(c for c in transport.command_calls if isinstance(c, str) and c.startswith("mkdir -p "))
    assert "/workspace/upscale-sessions/session-D-01/input" in prepare_cmd
    assert "/workspace/upscale-sessions/session-D-01/output" in prepare_cmd
    assert "/workspace/upscale-sessions/session-D-01/logs" in prepare_cmd
    assert transport.upload_calls[0][1] == "/workspace/upscale-sessions/session-D-01/input"
    worker_cmd = next(c for c in transport.command_calls if isinstance(c, list) and c[:2] == ["python", "/app/worker/process.py"])
    assert "--target-width" in worker_cmd
    assert "--target-height" in worker_cmd
    assert "--profile" in worker_cmd


def test_execute_remote_worker_non_zero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.process_result = RemoteCommandResult(returncode=1, stdout="", stderr="runtime failed")
    transport.report_payload = _completed_report("/workspace/upscale-sessions/session-D-01/output/result video.mp4")

    monkeypatch.setattr(
        "vast_upscaler.remote_worker.inspect_video",
        lambda *_a, **_k: VideoInfo(file_path="x", width=1920, height=1080, duration_seconds=10),
    )
    monkeypatch.setattr("vast_upscaler.remote_worker.run_ffprobe", lambda *_a, **_k: {"streams": []})

    with pytest.raises(RemoteWorkerProcessError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_report_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.fail_report_download = True

    monkeypatch.setattr("vast_upscaler.remote_worker.inspect_video", lambda *_a, **_k: None)

    with pytest.raises(RemoteWorkerReportError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_report_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.report_payload = b"{invalid"

    monkeypatch.setattr("vast_upscaler.remote_worker.inspect_video", lambda *_a, **_k: None)

    with pytest.raises(RemoteWorkerReportError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_report_failed_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.report_payload = json.dumps({"status": "failed", "frames_processed": 0, "output": "x"}).encode("utf-8")

    monkeypatch.setattr("vast_upscaler.remote_worker.inspect_video", lambda *_a, **_k: None)

    with pytest.raises(RemoteWorkerReportError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_output_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.report_payload = _completed_report("/workspace/upscale-sessions/session-D-01/output/result video.mp4")
    transport.fail_output_download = True

    monkeypatch.setattr("vast_upscaler.remote_worker.inspect_video", lambda *_a, **_k: None)

    with pytest.raises(RemoteWorkerOutputError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_resolution_incorrect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.report_payload = _completed_report("/workspace/upscale-sessions/session-D-01/output/result video.mp4")

    monkeypatch.setattr(
        "vast_upscaler.remote_worker.inspect_video",
        lambda *_a, **_k: VideoInfo(file_path="x", width=1280, height=720, duration_seconds=11.0),
    )

    with pytest.raises(RemoteWorkerOutputError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_corrupted_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.report_payload = _completed_report("/workspace/upscale-sessions/session-D-01/output/result video.mp4")

    def _raise_probe(*_a, **_k):
        raise VideoProbeError("corrupt")

    monkeypatch.setattr("vast_upscaler.remote_worker.inspect_video", _raise_probe)

    with pytest.raises(RemoteWorkerOutputError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_upload_transfer_failure(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.upload_error = TransferError("scp failed")

    with pytest.raises(RemoteWorkerTransferError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_prepare_dirs_failure(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.prepare_error = SshConnectionError("permission denied")

    with pytest.raises(RemoteWorkerError) as exc:
        execute_remote_worker(transport=transport, connection=conn, request=request)
    assert "Failed to prepare remote directories" in str(exc.value)


def test_execute_remote_worker_prepare_dirs_command_quotes_special_chars(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"src")
    request = RemoteWorkerRequest(
        session_id="sess odd [x]/v1",
        local_input_path=source,
        local_output_path=tmp_path / "out" / "clip [done].mp4",
        local_report_path=tmp_path / "out" / "report.json",
        target_width=1920,
        target_height=1080,
    )
    transport = FakeTransport()
    transport.report_payload = _completed_report("/workspace/upscale-sessions/sess-odd-x-v1/output/clip [done].mp4")

    with pytest.raises(RemoteWorkerOutputError):
        execute_remote_worker(transport=transport, connection=conn, request=request)

    prepare_cmd = next(c for c in transport.command_calls if isinstance(c, str) and c.startswith("mkdir -p "))
    assert "/workspace/upscale-sessions/sess-odd-x-v1/input" in prepare_cmd
    assert "/workspace/upscale-sessions/sess-odd-x-v1/output" in prepare_cmd
    assert "/workspace/upscale-sessions/sess-odd-x-v1/logs" in prepare_cmd


def test_execute_remote_worker_timeout_ambiguous_state(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.process_error = SshTimeoutError("timeout")
    transport.probe_result = RemoteCommandResult(returncode=0, stdout="REPORT_PRESENT=1\nOUTPUT_PRESENT=0\n", stderr="")

    with pytest.raises(RemoteWorkerAmbiguousStateError):
        execute_remote_worker(transport=transport, connection=conn, request=request)


def test_execute_remote_worker_interruption_with_unknown_state(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    request = _request(tmp_path)
    transport = FakeTransport()
    transport.process_error = SshConnectionError("connection reset")
    transport.probe_error = SshConnectionError("probe failed")

    with pytest.raises(RemoteWorkerAmbiguousStateError):
        execute_remote_worker(transport=transport, connection=conn, request=request)
