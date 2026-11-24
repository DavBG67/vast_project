from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vast_upscaler.remote_seedvr2_worker import (
    RemoteSeedvr2WorkerProcessError,
    RemoteSeedvr2WorkerRecoveryError,
    RemoteSeedvr2WorkerReportError,
    Seedvr2RemoteRequest,
    _build_seedvr2_discovery_script,
    execute_remote_seedvr2_worker,
    recover_remote_seedvr2_results,
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
                    "log": "/workspace/seedvr2-sessions/manual-01/output/run/run.log",
                }
            ),
            stderr="",
        )
        self.report_payload = json.dumps(
            {
                "status": "completed",
                "output_file": "/workspace/seedvr2-sessions/manual-01/output/run/upscaled_h264_aac.mp4",
                "log_file": "/workspace/seedvr2-sessions/manual-01/output/run/run.log",
            }
        ).encode("utf-8")
        self.stream_lines: list[str] = []

    def upload_file(self, _conn, *, local_path, remote_dir, **_kwargs):
        self.upload_calls.append((Path(local_path), str(remote_dir)))
        return f"{remote_dir}/{Path(local_path).name}"

    def run_remote_command(self, _conn, *, command, **kwargs):
        self.command_calls.append(command)
        if isinstance(command, str) and command.startswith("mkdir -p "):
            return RemoteCommandResult(returncode=0, stdout="", stderr="")
        if isinstance(command, list) and command[:2] == ["/opt/seedvr2-venv/bin/python", "-c"]:
            script = command[2]
            if "urllib.request" in script:
                return RemoteCommandResult(returncode=0, stdout="", stderr="")
            return self.discovery_result
        if isinstance(command, str) and "/opt/prototype/seedvr2_prototype.py" in command:
            callback = kwargs.get("stdout_line_callback")
            if callback is not None:
                for line in self.stream_lines:
                    callback(line)
            return self.process_result
        raise AssertionError(f"Unexpected command: {command}")

    def download_file(self, _conn, *, remote_path, local_path, **_kwargs):
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(remote_path).endswith("prototype-report.json"):
            target.write_bytes(self.report_payload)
            return target
        if str(remote_path).endswith("/run.log"):
            target.write_text("seedvr2 log", encoding="utf-8")
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
    events: list[dict[str, object]] = []
    result = execute_remote_seedvr2_worker(
        transport=transport,
        connection=_connection(tmp_path),
        request=_request(tmp_path),
        progress_callback=events.append,
    )
    assert result.local_output_path.exists()
    assert result.local_report_path.exists()
    assert result.report_payload["status"] == "completed"
    assert result.local_log_path is not None
    assert result.local_log_path.exists()
    assert any(event.get("stage") == "checkpoint-cache" and event.get("status") == "completed" for event in events)

    worker_cmd = next(c for c in transport.command_calls if isinstance(c, str) and "/opt/prototype/seedvr2_prototype.py" in c)
    assert "2>&1" in worker_cmd


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


def test_execute_remote_seedvr2_worker_rejects_output_outside_output_root(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.discovery_result = RemoteCommandResult(
        returncode=0,
        stdout=json.dumps(
            {
                "report": "/workspace/seedvr2-sessions/manual-01/output/run/prototype-report.json",
                "output": "/tmp/escape.mp4",
                "log": "/workspace/seedvr2-sessions/manual-01/output/run/run.log",
            }
        ),
        stderr="",
    )
    with pytest.raises(RemoteSeedvr2WorkerReportError):
        execute_remote_seedvr2_worker(transport=transport, connection=_connection(tmp_path), request=_request(tmp_path))


def test_execute_remote_seedvr2_worker_download_failure_is_recovery_error(tmp_path: Path) -> None:
    class FailingTransport(FakeTransport):
        def download_file(self, _conn, *, remote_path, local_path, **_kwargs):
            if str(remote_path).endswith("prototype-report.json"):
                raise TransferError("report missing")
            return super().download_file(_conn, remote_path=remote_path, local_path=local_path, **_kwargs)

    with pytest.raises(RemoteSeedvr2WorkerRecoveryError):
        execute_remote_seedvr2_worker(
            transport=FailingTransport(),
            connection=_connection(tmp_path),
            request=_request(tmp_path),
        )


def test_recover_remote_seedvr2_results_downloads_without_reinference(tmp_path: Path) -> None:
    transport = FakeTransport()
    request = _request(tmp_path)
    result = recover_remote_seedvr2_results(transport=transport, connection=_connection(tmp_path), request=request)
    assert result.local_output_path.exists()
    assert result.local_report_path.exists()
    runner_calls = [c for c in transport.command_calls if isinstance(c, str) and "/opt/prototype/seedvr2_prototype.py" in c]
    assert runner_calls == []


def test_discovery_script_regression_reproduces_old_syntax_error() -> None:
    old_script = (
        "import json; from pathlib import Path; root = Path('/tmp'); "
        "reports = []; if reports: print('x')"
    )
    run = subprocess.run([sys.executable, "-c", old_script], capture_output=True, text=True, check=False)
    assert run.returncode != 0
    assert "SyntaxError" in (run.stderr or "")


def test_discovery_script_executes_with_spaces_and_without_log(tmp_path: Path) -> None:
    output_root = tmp_path / "seed output"
    run_dir = output_root / "run 01"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_file = run_dir / "upscaled_h264_aac.mp4"
    output_file.write_bytes(b"video")
    report = run_dir / "prototype-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "output_file": str(output_file),
                "log_file": str(run_dir / "missing.log"),
            }
        ),
        encoding="utf-8",
    )

    script = _build_seedvr2_discovery_script(output_root=str(output_root))
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert run.returncode == 0
    payload = json.loads(run.stdout.strip())
    assert Path(payload["report"]).name == "prototype-report.json"
    assert Path(payload["output"]).name == "upscaled_h264_aac.mp4"
    assert payload["log"] is None


def test_discovery_script_rejects_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "root"
    run_dir = output_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    link = run_dir / "output-link.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks are not available in this environment")

    report = run_dir / "prototype-report.json"
    report.write_text(
        json.dumps({"status": "completed", "output_file": str(link)}),
        encoding="utf-8",
    )

    script = _build_seedvr2_discovery_script(output_root=str(output_root))
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert run.returncode != 0
    assert "outside output root" in (run.stderr or "")
