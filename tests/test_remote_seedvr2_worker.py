from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import pytest

from vast_upscaler.remote_seedvr2_worker import (
    RemoteSeedvr2WorkerPreparationError,
    RemoteSeedvr2WorkerProcessError,
    RemoteSeedvr2WorkerRecoveryError,
    Seedvr2RemoteRequest,
    _build_checkpoint_cache_script,
    _build_seedvr2_discovery_script,
    _parse_seedvr2_progress_line,
    _Seedvr2ProgressState,
    execute_remote_seedvr2_worker,
    recover_remote_seedvr2_results,
)
from vast_upscaler.transfer import (
    RemoteCommandResult,
    SshConnectionInfo,
    SshTransportError,
    TransferError,
)


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


def test_execute_remote_seedvr2_worker_failed_report_status_is_inference_error(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.report_payload = json.dumps({"status": "failed"}).encode("utf-8")
    with pytest.raises(RemoteSeedvr2WorkerProcessError):
        execute_remote_seedvr2_worker(transport=transport, connection=_connection(tmp_path), request=_request(tmp_path))


def test_execute_remote_seedvr2_worker_invalid_discovery_json(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.discovery_result = RemoteCommandResult(returncode=0, stdout="not-json", stderr="")
    with pytest.raises(RemoteSeedvr2WorkerRecoveryError):
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
    with pytest.raises(RemoteSeedvr2WorkerRecoveryError):
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
    except OSError as exc:
        if os.name == "nt":
            pytest.skip("symlink creation is not permitted in this Windows environment")
        raise AssertionError("symlink creation should be available in CI Linux") from exc

    report = run_dir / "prototype-report.json"
    report.write_text(
        json.dumps({"status": "completed", "output_file": str(link)}),
        encoding="utf-8",
    )

    script = _build_seedvr2_discovery_script(output_root=str(output_root))
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert run.returncode != 0
    assert "outside output root" in (run.stderr or "")


def test_seedvr2_progress_parser_uses_expected_patterns() -> None:
    state = _Seedvr2ProgressState()

    assert _parse_seedvr2_progress_line(line="VAE encode started", state=state) == {
        "kind": "worker",
        "stage": "vae-encode",
        "determinate": False,
    }
    assert _parse_seedvr2_progress_line(line="DiT upscale running", state=state) == {
        "kind": "worker",
        "stage": "dit-upscale",
        "determinate": False,
    }
    assert _parse_seedvr2_progress_line(line="chunk 1/4", state=state) == {
        "kind": "worker",
        "stage": "dit-upscale",
        "chunk_current": 1,
        "chunk_total": 4,
        "chunks_completed": 0,
        "progress": pytest.approx(0.0),
        "determinate": False,
    }
    frame_event = _parse_seedvr2_progress_line(line="frames 25/100", state=state)
    assert frame_event is not None
    assert frame_event["frames_done"] == 25
    assert frame_event["frames_total"] == 100
    assert frame_event["progress"] == pytest.approx(0.25)
    assert _parse_seedvr2_progress_line(line="unrecognized noise line", state=state) is None


def test_seedvr2_progress_parser_keeps_global_progress_non_decreasing() -> None:
    state = _Seedvr2ProgressState()
    first = _parse_seedvr2_progress_line(line="frames 60/100", state=state)
    second = _parse_seedvr2_progress_line(line="frames 10/100", state=state)
    assert first is not None and second is not None
    assert first["progress"] == pytest.approx(0.6)
    assert second["progress"] == pytest.approx(0.6)


def test_seedvr2_progress_parser_does_not_treat_chunk_internal_counters_as_global() -> None:
    state = _Seedvr2ProgressState()
    event = _parse_seedvr2_progress_line(line="chunk 2/8 - frames 4/32", state=state)
    assert event == {
        "kind": "worker",
        "stage": "dit-upscale",
        "chunk_current": 2,
        "chunk_total": 8,
        "chunks_completed": 1,
        "progress": pytest.approx(0.125),
        "determinate": False,
    }
    assert state.last_progress == pytest.approx(0.125)


def test_execute_remote_seedvr2_worker_emits_progress_before_inference_end(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.stream_lines = [
        "VAE encode",
        "chunk 1/2",
        "frames 10/100",
        "DiT upscale",
    ]
    events: list[dict[str, object]] = []
    execute_remote_seedvr2_worker(
        transport=transport,
        connection=_connection(tmp_path),
        request=_request(tmp_path),
        progress_callback=events.append,
    )
    frame_event_index = next(i for i, payload in enumerate(events) if payload.get("frames_done") == 10)
    download_started_index = next(
        i for i, payload in enumerate(events) if payload.get("kind") == "transfer" and payload.get("stage") == "download"
    )
    assert frame_event_index < download_started_index


def test_execute_remote_seedvr2_worker_cache_prep_error_is_preparation_error(tmp_path: Path) -> None:
    class FailingCacheTransport(FakeTransport):
        def run_remote_command(self, _conn, *, command, **kwargs):
            if isinstance(command, list) and command[:2] == ["/opt/seedvr2-venv/bin/python", "-c"]:
                script = command[2]
                if "urllib.request" in script:
                    raise SshTransportError("cache prep failed")
            return super().run_remote_command(_conn, command=command, **kwargs)

    with pytest.raises(RemoteSeedvr2WorkerPreparationError):
        execute_remote_seedvr2_worker(
            transport=FailingCacheTransport(),
            connection=_connection(tmp_path),
            request=_request(tmp_path),
        )


def _run_python_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)


def _checkpoint_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX flock semantics")
def test_checkpoint_cache_script_handles_concurrency_and_reuse(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    checkpoint = source_dir / "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    vae = source_dir / "ema_vae_fp16.safetensors"
    checkpoint.write_bytes(b"checkpoint-v1")
    vae.write_bytes(b"vae-v1")
    metadata = {
        checkpoint.name: {"repo": "local/test", "sha256": _checkpoint_sha(checkpoint), "url": checkpoint.as_uri()},
        vae.name: {"repo": "local/test", "sha256": _checkpoint_sha(vae), "url": vae.as_uri()},
    }
    script = _build_checkpoint_cache_script(cache_dir=str(cache_dir), metadata_override=metadata, hold_lock_seconds=1.0)
    proc_a = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.2)
    proc_b = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out_a, err_a = proc_a.communicate(timeout=10)
    out_b, err_b = proc_b.communicate(timeout=10)
    assert proc_a.returncode == 0, err_a or out_a
    assert proc_b.returncode == 0, err_b or out_b
    assert (cache_dir / checkpoint.name).exists()
    assert (cache_dir / vae.name).exists()

    checkpoint.write_bytes(b"checkpoint-v2")
    vae.write_bytes(b"vae-v2")
    metadata_updated = {
        checkpoint.name: {"repo": "local/test", "sha256": _checkpoint_sha(checkpoint), "url": checkpoint.as_uri()},
        vae.name: {"repo": "local/test", "sha256": _checkpoint_sha(vae), "url": vae.as_uri()},
    }
    rerun = _run_python_script(_build_checkpoint_cache_script(cache_dir=str(cache_dir), metadata_override=metadata_updated))
    assert rerun.returncode == 0, rerun.stderr
    assert (cache_dir / checkpoint.name).read_bytes() == b"checkpoint-v2"
    assert (cache_dir / vae.name).read_bytes() == b"vae-v2"


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX flock semantics")
def test_checkpoint_cache_script_lock_released_after_holder_termination(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    checkpoint = source_dir / "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    vae = source_dir / "ema_vae_fp16.safetensors"
    checkpoint.write_bytes(b"checkpoint-v1")
    vae.write_bytes(b"vae-v1")
    metadata = {
        checkpoint.name: {"repo": "local/test", "sha256": _checkpoint_sha(checkpoint), "url": checkpoint.as_uri()},
        vae.name: {"repo": "local/test", "sha256": _checkpoint_sha(vae), "url": vae.as_uri()},
    }
    long_script = _build_checkpoint_cache_script(
        cache_dir=str(cache_dir),
        metadata_override=metadata,
        lock_timeout_seconds=5.0,
        hold_lock_seconds=30.0,
    )
    holder = subprocess.Popen([sys.executable, "-c", long_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    holder.terminate()
    holder.communicate(timeout=10)
    retry = _run_python_script(
        _build_checkpoint_cache_script(
            cache_dir=str(cache_dir),
            metadata_override=metadata,
            lock_timeout_seconds=5.0,
        )
    )
    assert retry.returncode == 0, retry.stderr


def test_checkpoint_cache_script_cleans_part_file_after_download_failure(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    metadata = {
        "seedvr2_ema_3b_fp8_e4m3fn.safetensors": {
            "repo": "local/test",
            "sha256": "0" * 64,
            "url": "file:///definitely-missing-file",
        }
    }
    run = _run_python_script(
        _build_checkpoint_cache_script(
            cache_dir=str(cache_dir),
            metadata_override=metadata,
            lock_timeout_seconds=1.0,
        )
    )
    assert run.returncode != 0
    assert not (cache_dir / "seedvr2_ema_3b_fp8_e4m3fn.safetensors.part").exists()
