from __future__ import annotations

import json
from pathlib import Path

import pytest

from vast_upscaler.remote_seedvr2_worker import (
    RemoteSeedvr2WorkerProcessError,
    RemoteSeedvr2WorkerReportError,
    RemoteSeedvr2WorkerTransferError,
    Seedvr2RemoteRequest,
    execute_remote_seedvr2_worker,
)
from vast_upscaler.transfer import RemoteCommandResult, SshConnectionInfo, TransferError


class FakeTransport:
    def __init__(self) -> None:
        self.upload_calls: list[tuple[Path, str]] = []
        self.command_calls: list[list[str] | str] = []
        self.process_result = RemoteCommandResult(returncode=0, stdout="ok", stderr="")
        self.discovery_result = RemoteCommandResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "report": "/workspace/seedvr2-sessions/manual-01/output/run/prototype-report.json",
                    "output": "/workspace/seedvr2-sessions/manual-01/output/run/upscaled_h264_aac.mp4",
                }
            ),
            stderr="",
        )
        self.report_payload = json.dumps({"status": "completed", "output_file": "/tmp/upscaled.mp4"}).encode("utf-8")

    def upload_file(self, _conn, *, local_path, remote_dir, **_kwargs):
        self.upload_calls.append((Path(local_path), str(remote_dir)))
        return f"{remote_dir}/{Path(local_path).name}"

    def run_remote_command(self, _conn, *, command, **_kwargs):
        self.command_calls.append(command)
        if isinstance(command, str) and command.startswith("mkdir -p "):
            return RemoteCommandResult(returncode=0, stdout="", stderr="")
        if isinstance(command, list) and command[:2] == ["python", "-c"]:
            return self.discovery_result
        if isinstance(command, list) and command[:2] == ["python", "/opt/prototype/seedvr2_prototype.py"]:
            return self.process_result
        raise AssertionError(f"Unexpected command: {command}")

    def download_file(self, _conn, *, remote_path, local_path, **_kwargs):
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(remote_path).endswith("prototype-report.json"):
            target.write_bytes(self.report_payload)
            return target
        target.write_bytes(b"video")
        return target


def _connection(tmp_path: Path) -> SshConnectionInfo:
    key = tmp_path / "id_ed25519"
    key.write_text("key", encoding="utf-8")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("host", encoding="utf-8")
    return SshConnectionInfo(
        host="1.2.3.4",
        port=2222,
        username="root",
        private_key_path=key,
        known_hosts_path=known_hosts,
        mode="proxy",
    )


def _request(tmp_path: Path) -> Seedvr2RemoteRequest:
    source = tmp_path / "input clip.mp4"
    source.write_bytes(b"source")
    return Seedvr2RemoteRequest(
        session_id="manual-01",
        local_input_path=source,
        local_output_path=tmp_path / "out" / "result.mp4",
        local_report_path=tmp_path / "out" / "report.json",
        profile="smoke",
    )


def test_execute_remote_seedvr2_worker_nominal(tmp_path: Path) -> None:
    transport = FakeTransport()
    result = execute_remote_seedvr2_worker(transport=transport, connection=_connection(tmp_path), request=_request(tmp_path))
    assert result.local_output_path.exists()
    assert result.local_report_path.exists()
    assert result.report_payload["status"] == "completed"
    worker_cmd = next(c for c in transport.command_calls if isinstance(c, list) and c[:2] == ["python", "/opt/prototype/seedvr2_prototype.py"])
    assert "--input-is-excerpt" in worker_cmd
    assert "--profile" in worker_cmd


def test_execute_remote_seedvr2_worker_non_zero_exit(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.process_result = RemoteCommandResult(returncode=2, stdout="", stderr="failed")
    with pytest.raises(RemoteSeedvr2WorkerProcessError):
        execute_remote_seedvr2_worker(transport=transport, connection=_connection(tmp_path), request=_request(tmp_path))


def test_execute_remote_seedvr2_worker_invalid_discovery_json(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.discovery_result = RemoteCommandResult(returncode=0, stdout="not-json", stderr="")
    with pytest.raises(RemoteSeedvr2WorkerReportError):
        execute_remote_seedvr2_worker(transport=transport, connection=_connection(tmp_path), request=_request(tmp_path))


def test_execute_remote_seedvr2_worker_report_download_failure(tmp_path: Path) -> None:
    class FailingTransport(FakeTransport):
        def download_file(self, _conn, *, remote_path, local_path, **_kwargs):
            if str(remote_path).endswith("prototype-report.json"):
                raise TransferError("report missing")
            return super().download_file(_conn, remote_path=remote_path, local_path=local_path, **_kwargs)

    with pytest.raises(RemoteSeedvr2WorkerTransferError):
        execute_remote_seedvr2_worker(
            transport=FailingTransport(),
            connection=_connection(tmp_path),
            request=_request(tmp_path),
        )
