from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from vast_upscaler.transfer import (
    IntegrityCheckError,
    LocalFileExistsError,
    OpenSshTransport,
    RetryPolicy,
    SshAuthenticationError,
    SshBinaryNotFoundError,
    SshConfigError,
    SshConnectionError,
    SshConnectionInfo,
    SshKnownHostsError,
    SshTimeoutError,
    TransferInterruptedError,
    build_ssh_connection_info,
    compute_transfer_timeout_seconds,
    sha256_file,
    verify_local_file,
)
from vast_upscaler.vast_write_client import InstanceSummary


def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ssh"], returncode=returncode, stdout=stdout, stderr=stderr)


def _conn(tmp_path: Path) -> SshConnectionInfo:
    key = tmp_path / "id_ed25519"
    key.write_text("key", encoding="utf-8")
    kh = tmp_path / "known_hosts"
    kh.write_text("host", encoding="utf-8")
    return SshConnectionInfo(
        host="1.2.3.4",
        port=2222,
        username="root",
        private_key_path=key,
        known_hosts_path=kh,
        mode="proxy",
    )


def test_run_remote_command_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _conn(tmp_path)

    def fake_runner(*args, **kwargs):
        return _cp(returncode=0, stdout="ok\n", stderr="")

    transport = OpenSshTransport(runner=fake_runner)
    result = transport.run_remote_command(
        conn,
        command=["echo", "ok"],
        timeout_seconds=5,
        retry_policy=RetryPolicy(max_attempts=1),
        idempotent=True,
    )
    assert result.returncode == 0
    assert result.stdout == "ok\n"


def test_run_remote_command_host_key_failure(tmp_path: Path) -> None:
    conn = _conn(tmp_path)

    def fake_runner(*args, **kwargs):
        return _cp(returncode=255, stderr="Host key verification failed.")

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(SshKnownHostsError):
        transport.run_remote_command(
            conn,
            command=["true"],
            timeout_seconds=5,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
        )


def test_run_remote_command_auth_failure(tmp_path: Path) -> None:
    conn = _conn(tmp_path)

    def fake_runner(*args, **kwargs):
        return _cp(returncode=255, stderr="Permission denied (publickey).")

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(SshAuthenticationError):
        transport.run_remote_command(
            conn,
            command=["true"],
            timeout_seconds=5,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
        )


def test_run_remote_command_timeout_retry_then_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    state = {"n": 0}

    def fake_runner(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))
        return _cp(returncode=0, stdout="ready\n")

    monkeypatch.setattr(OpenSshTransport, "_sleep", staticmethod(lambda _s: None))
    transport = OpenSshTransport(runner=fake_runner)
    result = transport.run_remote_command(
        conn,
        command=["true"],
        timeout_seconds=1,
        retry_policy=RetryPolicy(max_attempts=2, retry_delay_seconds=0.01),
        idempotent=True,
    )
    assert result.stdout == "ready\n"
    assert state["n"] == 2


def test_run_remote_command_retry_for_non_idempotent_refused(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    transport = OpenSshTransport(runner=lambda *a, **k: _cp(returncode=0))
    with pytest.raises(SshConfigError):
        transport.run_remote_command(
            conn,
            command=["touch", "/tmp/new"],
            timeout_seconds=5,
            retry_policy=RetryPolicy(max_attempts=2),
            idempotent=False,
        )


def test_download_atomic_and_integrity_success(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    dst = tmp_path / "out" / "video.mp4"
    payload = b"hello-world"
    expected_sha = hashlib.sha256(payload).hexdigest()

    def fake_runner(args, **kwargs):
        if args[0] == "scp":
            tmp_local = Path(args[-1])
            tmp_local.write_bytes(payload)
            return _cp(returncode=0)
        return _cp(returncode=0)

    transport = OpenSshTransport(runner=fake_runner)
    result = transport.download_file(
        conn,
        remote_path="/remote/video.mp4",
        local_path=dst,
        timeout_seconds=5,
        expected_size_bytes=len(payload),
        expected_sha256=expected_sha,
    )
    assert result == dst
    assert dst.read_bytes() == payload


def test_download_refuses_overwrite(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    dst = tmp_path / "existing.bin"
    dst.write_bytes(b"x")

    transport = OpenSshTransport(runner=lambda *a, **k: _cp(returncode=0))
    with pytest.raises(LocalFileExistsError):
        transport.download_file(
            conn,
            remote_path="/remote/file.bin",
            local_path=dst,
            timeout_seconds=5,
            overwrite=False,
        )


def test_download_refuses_late_overwrite_before_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    dst = tmp_path / "late.bin"

    def fake_runner(args, **kwargs):
        if args[0] == "scp":
            Path(args[-1]).write_bytes(b"new")
        return _cp(returncode=0)

    original_verify = __import__("vast_upscaler.transfer", fromlist=["verify_local_file"]).verify_local_file

    def fake_verify(path: Path, *, expected_size_bytes=None, expected_sha256=None):
        dst.write_bytes(b"existing")
        original_verify(path, expected_size_bytes=expected_size_bytes, expected_sha256=expected_sha256)

    monkeypatch.setattr("vast_upscaler.transfer.verify_local_file", fake_verify)

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(LocalFileExistsError):
        transport.download_file(
            conn,
            remote_path="/remote/file.bin",
            local_path=dst,
            timeout_seconds=5,
            overwrite=False,
        )

    assert dst.read_bytes() == b"existing"
    assert list(tmp_path.glob("*.part")) == []


def test_download_cleans_partial_file_on_failure(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    dst = tmp_path / "out.bin"

    def fake_runner(*args, **kwargs):
        return _cp(returncode=255, stderr="connection reset by peer")

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(TransferInterruptedError):
        transport.download_file(
            conn,
            remote_path="/remote/file.bin",
            local_path=dst,
            timeout_seconds=5,
            retry_policy=RetryPolicy(max_attempts=1),
        )
    parts = list(dst.parent.glob(".*.part"))
    assert parts == []


def test_upload_windows_path_with_spaces(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    local = tmp_path / "my file [v1].txt"
    local.write_text("abc", encoding="utf-8")

    calls = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        return _cp(returncode=0)

    transport = OpenSshTransport(runner=fake_runner)
    remote = transport.upload_file(
        conn,
        local_path=local,
        remote_dir="/remote data",
        timeout_seconds=5,
    )
    assert remote.endswith("my file [v1].txt")
    assert calls
    assert calls[0][0] == "scp"
    assert str(local) in calls[0]


def test_upload_retry_non_idempotent_transfer_refused(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    local = tmp_path / "in.bin"
    local.write_bytes(b"data")

    transport = OpenSshTransport(runner=lambda *a, **k: _cp(returncode=0))
    with pytest.raises(SshConfigError):
        transport.upload_file(
            conn,
            local_path=local,
            remote_dir="/remote",
            timeout_seconds=5,
            retry_policy=RetryPolicy(max_attempts=2),
        )


def test_build_ssh_connection_info_valid(tmp_path: Path) -> None:
    key = tmp_path / "id"
    key.write_text("key", encoding="utf-8")
    kh = tmp_path / "known_hosts"
    kh.write_text("host", encoding="utf-8")

    instance = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="test",
        ssh_host="5.6.7.8",
        ssh_port=2200,
        image_runtype="ssh proxy",
    )
    info = build_ssh_connection_info(
        instance=instance,
        private_key_path=key,
        known_hosts_path=kh,
    )
    assert info.host == "5.6.7.8"
    assert info.port == 2200
    assert info.mode == "proxy"


def test_build_ssh_connection_info_requires_known_hosts(tmp_path: Path) -> None:
    key = tmp_path / "id"
    key.write_text("key", encoding="utf-8")

    instance = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="test",
        ssh_host="5.6.7.8",
        ssh_port=2200,
        image_runtype=None,
    )
    with pytest.raises(SshKnownHostsError):
        build_ssh_connection_info(
            instance=instance,
            private_key_path=key,
            known_hosts_path=tmp_path / "missing_known_hosts",
        )


def test_exec_binary_not_found_maps_error(tmp_path: Path) -> None:
    conn = _conn(tmp_path)

    def fake_runner(*args, **kwargs):
        raise FileNotFoundError("ssh")

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(SshBinaryNotFoundError):
        transport.run_remote_command(
            conn,
            command=["true"],
            timeout_seconds=3,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
        )


def test_compute_transfer_timeout_seconds() -> None:
    assert compute_transfer_timeout_seconds(expected_size_bytes=None) == 60.0
    assert compute_transfer_timeout_seconds(expected_size_bytes=1000) == 60.0
    assert compute_transfer_timeout_seconds(expected_size_bytes=20_000_000, minimum_seconds=5.0) >= 5.0


def test_sha_and_verify_helpers(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")

    digest = sha256_file(f)
    assert digest == hashlib.sha256(b"abc").hexdigest()

    verify_local_file(f, expected_size_bytes=3, expected_sha256=digest)

    with pytest.raises(IntegrityCheckError):
        verify_local_file(f, expected_size_bytes=4)

    with pytest.raises(IntegrityCheckError):
        verify_local_file(f, expected_sha256="deadbeef")


def test_download_integrity_failure_removes_tmp(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    dst = tmp_path / "artifact.bin"

    def fake_runner(args, **kwargs):
        if args[0] == "scp":
            Path(args[-1]).write_bytes(b"short")
        return _cp(returncode=0)

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(IntegrityCheckError):
        transport.download_file(
            conn,
            remote_path="/remote/artifact.bin",
            local_path=dst,
            timeout_seconds=5,
            expected_size_bytes=123,
        )
    assert not dst.exists()


def test_connection_error_classification(tmp_path: Path) -> None:
    conn = _conn(tmp_path)

    def fake_runner(*args, **kwargs):
        return _cp(returncode=255, stderr="ssh: connect to host 1.2.3.4 port 2222: Connection refused")

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(SshConnectionError):
        transport.run_remote_command(
            conn,
            command=["true"],
            timeout_seconds=5,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
        )


def test_run_remote_command_timeout_raises(tmp_path: Path) -> None:
    conn = _conn(tmp_path)

    def fake_runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)

    transport = OpenSshTransport(runner=fake_runner)
    with pytest.raises(SshTimeoutError):
        transport.run_remote_command(
            conn,
            command=["true"],
            timeout_seconds=1,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
        )
