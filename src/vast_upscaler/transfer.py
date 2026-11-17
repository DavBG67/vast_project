"""Windows-compatible SSH/SCP transport layer (phase 4C)."""

from __future__ import annotations

import hashlib
import os
import posixpath
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vast_upscaler.vast_write_client import InstanceSummary


class SshTransportError(RuntimeError):
    pass


class SshConfigError(SshTransportError):
    pass


class SshBinaryNotFoundError(SshTransportError):
    pass


class SshKeyNotFoundError(SshTransportError):
    pass


class SshKnownHostsError(SshTransportError):
    pass


class SshAuthenticationError(SshTransportError):
    pass


class SshConnectionError(SshTransportError):
    pass


class SshTimeoutError(SshTransportError):
    pass


class RemoteCommandError(SshTransportError):
    pass


class TransferError(SshTransportError):
    pass


class TransferInterruptedError(TransferError):
    pass


class LocalFileExistsError(TransferError):
    pass


class LocalFileMissingError(TransferError):
    pass


class IntegrityCheckError(TransferError):
    pass


@dataclass(frozen=True)
class SshConnectionInfo:
    host: str
    port: int
    username: str
    private_key_path: Path
    known_hosts_path: Path
    mode: Literal["direct", "proxy", "unknown"]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retry_delay_seconds: float = 1.0


@dataclass(frozen=True)
class RemoteCommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenSshTransport:
    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner or subprocess.run

    def check_ssh_ready(
        self,
        conn: SshConnectionInfo,
        *,
        timeout_seconds: float,
        retry_policy: RetryPolicy | None = None,
    ) -> RemoteCommandResult:
        return self.run_remote_command(
            conn,
            command=["true"],
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy or RetryPolicy(max_attempts=3, retry_delay_seconds=2.0),
            idempotent=True,
            check=True,
        )

    def run_remote_command(
        self,
        conn: SshConnectionInfo,
        *,
        command: list[str] | str,
        timeout_seconds: float,
        retry_policy: RetryPolicy | None = None,
        idempotent: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        policy = retry_policy or RetryPolicy()
        if policy.max_attempts < 1:
            raise SshConfigError("retry_policy.max_attempts must be >= 1")
        if policy.max_attempts > 1 and not idempotent:
            raise SshConfigError("Refusing retries for non-idempotent remote command.")

        remote_cmd = command if isinstance(command, str) else " ".join(shlex.quote(part) for part in command)
        args = self._build_ssh_args(conn) + [remote_cmd]

        last_exc: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                completed = self._exec(args, timeout_seconds=timeout_seconds)
            except (SshConnectionError, SshTimeoutError) as exc:
                last_exc = exc
                if attempt < policy.max_attempts:
                    self._sleep(policy.retry_delay_seconds)
                    continue
                raise

            result = RemoteCommandResult(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            if not check or completed.returncode == 0:
                return result
            raise self._classify_process_error(completed, is_transfer=False)

        if last_exc is not None:
            raise last_exc
        raise SshTransportError("Remote command failed without a classified error.")

    def upload_file(
        self,
        conn: SshConnectionInfo,
        *,
        local_path: Path,
        remote_dir: str,
        remote_filename: str | None = None,
        timeout_seconds: float,
        retry_policy: RetryPolicy | None = None,
        verify_remote_sha256: bool = False,
    ) -> str:
        src = Path(local_path)
        if not src.exists():
            raise LocalFileMissingError(f"Local source file not found: {src}")

        name = remote_filename or src.name
        remote_path = posixpath.join(remote_dir, name)
        target = _remote_target(conn, remote_path)
        args = self._build_scp_args(conn) + [str(src), target]

        self._run_transfer(args, timeout_seconds=timeout_seconds, retry_policy=retry_policy, idempotent=False)

        if verify_remote_sha256:
            local_sha = sha256_file(src)
            result = self.run_remote_command(
                conn,
                command=["sha256sum", remote_path],
                timeout_seconds=timeout_seconds,
                retry_policy=RetryPolicy(max_attempts=1),
                idempotent=True,
                check=True,
            )
            remote_sha = result.stdout.strip().split()[0] if result.stdout.strip() else ""
            if remote_sha.lower() != local_sha.lower():
                raise IntegrityCheckError("Remote SHA-256 does not match local file checksum.")

        return remote_path

    def download_file(
        self,
        conn: SshConnectionInfo,
        *,
        remote_path: str,
        local_path: Path,
        timeout_seconds: float,
        retry_policy: RetryPolicy | None = None,
        overwrite: bool = False,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        dst = Path(local_path)
        if dst.exists() and not overwrite:
            raise LocalFileExistsError(f"Refusing to overwrite existing local file: {dst}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".part", dir=str(dst.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            src = _remote_target(conn, remote_path)
            args = self._build_scp_args(conn) + [src, str(tmp_path)]
            self._run_transfer(args, timeout_seconds=timeout_seconds, retry_policy=retry_policy, idempotent=True)

            verify_local_file(tmp_path, expected_size_bytes=expected_size_bytes, expected_sha256=expected_sha256)

            if dst.exists() and not overwrite:
                raise LocalFileExistsError(f"Refusing to overwrite existing local file: {dst}")
            os.replace(tmp_path, dst)
            return dst
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _run_transfer(
        self,
        args: list[str],
        *,
        timeout_seconds: float,
        retry_policy: RetryPolicy | None,
        idempotent: bool,
    ) -> None:
        policy = retry_policy or RetryPolicy()
        if policy.max_attempts < 1:
            raise SshConfigError("retry_policy.max_attempts must be >= 1")
        if policy.max_attempts > 1 and not idempotent:
            raise SshConfigError("Refusing retries for non-idempotent transfer.")

        last_exc: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                completed = self._exec(args, timeout_seconds=timeout_seconds)
            except (SshConnectionError, SshTimeoutError) as exc:
                last_exc = exc
                if attempt < policy.max_attempts:
                    self._sleep(policy.retry_delay_seconds)
                    continue
                raise

            if completed.returncode == 0:
                return

            exc = self._classify_process_error(completed, is_transfer=True)
            last_exc = exc
            if isinstance(exc, TransferInterruptedError) and attempt < policy.max_attempts:
                self._sleep(policy.retry_delay_seconds)
                continue
            raise exc

        if last_exc is not None:
            raise last_exc
        raise TransferError("Transfer failed without a classified error.")

    def _build_ssh_args(self, conn: SshConnectionInfo) -> list[str]:
        return [
            "ssh",
            "-i",
            str(conn.private_key_path),
            "-p",
            str(conn.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={conn.known_hosts_path}",
            "-o",
            "NumberOfPasswordPrompts=0",
            f"{conn.username}@{conn.host}",
            "--",
        ]

    def _build_scp_args(self, conn: SshConnectionInfo) -> list[str]:
        return [
            "scp",
            "-i",
            str(conn.private_key_path),
            "-P",
            str(conn.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={conn.known_hosts_path}",
            "-o",
            "NumberOfPasswordPrompts=0",
        ]

    def _exec(self, args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                args,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except FileNotFoundError as exc:
            tool = args[0] if args else "ssh/scp"
            raise SshBinaryNotFoundError(f"OpenSSH binary not found: {tool}") from exc
        except subprocess.TimeoutExpired as exc:
            raise SshTimeoutError(f"OpenSSH command timed out after {timeout_seconds:.1f}s") from exc

    @staticmethod
    def _sleep(seconds: float) -> None:
        import time

        time.sleep(seconds)

    def _classify_process_error(self, completed: subprocess.CompletedProcess[str], *, is_transfer: bool) -> Exception:
        stderr = (completed.stderr or "").lower()

        if "host key verification failed" in stderr or "remote host identification has changed" in stderr:
            return SshKnownHostsError(completed.stderr.strip() or "SSH host key verification failed")
        if "no matching host key type found" in stderr or "offending" in stderr:
            return SshKnownHostsError(completed.stderr.strip() or "SSH host key mismatch")
        if "permission denied" in stderr:
            return SshAuthenticationError(completed.stderr.strip() or "SSH authentication failed")
        if "connection refused" in stderr or "connection timed out" in stderr or "could not resolve hostname" in stderr:
            if is_transfer:
                return TransferInterruptedError(completed.stderr.strip() or "SCP connectivity failure")
            return SshConnectionError(completed.stderr.strip() or "SSH connection failure")
        if "connection reset" in stderr or "broken pipe" in stderr:
            if is_transfer:
                return TransferInterruptedError(completed.stderr.strip() or "SCP transfer interrupted")
            return SshConnectionError(completed.stderr.strip() or "SSH connection interrupted")

        if is_transfer:
            return TransferError(completed.stderr.strip() or f"SCP failed with return code {completed.returncode}")
        return RemoteCommandError(completed.stderr.strip() or f"Remote command failed with return code {completed.returncode}")


def build_ssh_connection_info(
    *,
    instance: InstanceSummary,
    private_key_path: Path,
    known_hosts_path: Path,
    username: str = "root",
) -> SshConnectionInfo:
    if not instance.ssh_host or instance.ssh_port is None:
        raise SshConfigError("Instance does not expose SSH host/port yet.")

    key_path = Path(private_key_path)
    if not key_path.exists():
        raise SshKeyNotFoundError(f"SSH private key not found: {key_path}")

    kh_path = Path(known_hosts_path)
    if not kh_path.exists():
        raise SshKnownHostsError(
            f"Known hosts file not found: {kh_path}. "
            "Strict host key checking is mandatory; establish trust explicitly before connecting."
        )

    runtype = (instance.image_runtype or "").lower()
    if "direct" in runtype:
        mode: Literal["direct", "proxy", "unknown"] = "direct"
    elif "proxy" in runtype or "ssh" in runtype:
        mode = "proxy"
    else:
        mode = "unknown"

    return SshConnectionInfo(
        host=instance.ssh_host,
        port=instance.ssh_port,
        username=username,
        private_key_path=key_path,
        known_hosts_path=kh_path,
        mode=mode,
    )


def compute_transfer_timeout_seconds(
    *,
    expected_size_bytes: int | None,
    minimum_seconds: float = 60.0,
    bytes_per_second_floor: float = 2_000_000.0,
    headroom_factor: float = 2.0,
) -> float:
    if expected_size_bytes is None or expected_size_bytes <= 0:
        return minimum_seconds
    estimated = (expected_size_bytes / max(bytes_per_second_floor, 1.0)) * headroom_factor
    return max(minimum_seconds, estimated)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_file(
    path: Path,
    *,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    p = Path(path)
    if not p.exists():
        raise IntegrityCheckError(f"File missing after transfer: {p}")

    if expected_size_bytes is not None:
        actual_size = p.stat().st_size
        if actual_size != expected_size_bytes:
            raise IntegrityCheckError(
                f"Size mismatch: expected {expected_size_bytes} bytes, got {actual_size} bytes"
            )

    if expected_sha256 is not None:
        actual_sha = sha256_file(p)
        if actual_sha.lower() != expected_sha256.lower():
            raise IntegrityCheckError("SHA-256 mismatch for transferred file")


def _remote_target(conn: SshConnectionInfo, remote_path: str) -> str:
    return f"{conn.username}@{conn.host}:{_quote_remote_path(remote_path)}"


def _quote_remote_path(path: str) -> str:
    return "'" + path.replace("'", "'\"'\"'") + "'"
