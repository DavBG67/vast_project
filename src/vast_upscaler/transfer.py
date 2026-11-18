"""Windows-compatible SSH/SCP transport layer (phase 4C)."""

from __future__ import annotations

import base64
import hashlib
import os
import posixpath
import shlex
import shutil
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
class KnownHostKey:
    marker: str
    key_type: str
    key_data: str
    sha256_fingerprint: str
    source_line: str


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

_PREFERRED_KEX_CANDIDATES = [
    "sntrup761x25519-sha512@openssh.com",
    "curve25519-sha256",
    "curve25519-sha256@libssh.org",
    "ecdh-sha2-nistp256",
    "diffie-hellman-group14-sha256",
]


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
        args = [
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
        ]
        kex_algorithms = preferred_kex_algorithms_for_local_ssh()
        if kex_algorithms:
            args.extend(
                [
                    "-o",
                    f"KexAlgorithms={','.join(kex_algorithms)}",
                ]
            )
        args.extend(
            [
            f"{conn.username}@{conn.host}",
            "--",
            ]
        )
        return args

    def _build_scp_args(self, conn: SshConnectionInfo) -> list[str]:
        args = [
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
        kex_algorithms = preferred_kex_algorithms_for_local_ssh()
        if kex_algorithms:
            args.extend(
                [
                    "-o",
                    f"KexAlgorithms={','.join(kex_algorithms)}",
                ]
            )
        return args

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


def validate_local_ssh_prerequisites(*, private_key_path: Path, known_hosts_path: Path) -> None:
    for tool in ("ssh", "scp", "ssh-keyscan"):
        if shutil.which(tool) is None:
            raise SshBinaryNotFoundError(f"OpenSSH binary not found in PATH: {tool}")

    key_path = Path(private_key_path)
    if not key_path.exists():
        raise SshKeyNotFoundError(f"SSH private key not found: {key_path}")

    kh_path = Path(known_hosts_path)
    kh_path.parent.mkdir(parents=True, exist_ok=True)
    if not kh_path.exists():
        kh_path.write_text("", encoding="utf-8")


def known_host_markers(host: str, port: int) -> list[str]:
    markers = [f"[{host}]:{port}"]
    if port == 22:
        markers.append(host)
    return markers


def lookup_known_host_key(*, known_hosts_path: Path, host: str, port: int) -> KnownHostKey | None:
    kh_path = Path(known_hosts_path)
    if not kh_path.exists():
        return None

    marker_set = set(known_host_markers(host, port))
    for line in kh_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        marker, key_type, key_data = parts[0], parts[1], parts[2]
        if marker not in marker_set:
            continue
        if key_type != "ssh-ed25519":
            continue
        return KnownHostKey(
            marker=marker,
            key_type=key_type,
            key_data=key_data,
            sha256_fingerprint=_fingerprint_from_key_data(key_data),
            source_line=stripped,
        )
    return None


def fetch_ssh_ed25519_host_key(*, host: str, port: int, timeout_seconds: float = 15.0) -> KnownHostKey:
    marker = f"[{host}]:{port}"
    try:
        completed = subprocess.run(
            ["ssh-keyscan", "-p", str(port), "-t", "ed25519", host],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise SshBinaryNotFoundError("OpenSSH binary not found in PATH: ssh-keyscan") from exc
    except subprocess.TimeoutExpired as exc:
        raise SshConnectionError(f"ssh-keyscan timed out after {timeout_seconds:.1f}s") from exc

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip() and not line.startswith("#")]
    if completed.returncode != 0 and not lines:
        if _is_unsupported_kex_error(completed.stderr):
            fallback = _fetch_ssh_ed25519_host_key_with_ssh(
                host=host,
                port=port,
                timeout_seconds=timeout_seconds,
            )
            if fallback is not None:
                return fallback
        stderr = completed.stderr.strip() or "ssh-keyscan failed"
        raise SshConnectionError(stderr)
    if not lines:
        raise SshConnectionError("ssh-keyscan returned no host key")

    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        key_marker, key_type, key_data = parts[0], parts[1], parts[2]
        if key_type != "ssh-ed25519":
            continue
        normalized_marker = key_marker if key_marker else marker
        return KnownHostKey(
            marker=normalized_marker,
            key_type=key_type,
            key_data=key_data,
            sha256_fingerprint=_fingerprint_from_key_data(key_data),
            source_line=f"{normalized_marker} {key_type} {key_data}",
        )
    raise SshConnectionError("ssh-keyscan did not return an ED25519 host key")


def add_known_host_key(*, known_hosts_path: Path, host_key: KnownHostKey) -> None:
    kh_path = Path(known_hosts_path)
    kh_path.parent.mkdir(parents=True, exist_ok=True)
    if not kh_path.exists():
        kh_path.write_text("", encoding="utf-8")

    with kh_path.open("a", encoding="utf-8") as handle:
        if kh_path.stat().st_size > 0:
            handle.write("\n")
        handle.write(host_key.source_line)


def local_ssh_supported_kex_algorithms() -> set[str]:
    try:
        completed = subprocess.run(
            ["ssh", "-Q", "kex"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    if completed.returncode != 0:
        return set()
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def preferred_kex_algorithms_for_local_ssh() -> list[str]:
    supported = local_ssh_supported_kex_algorithms()
    return [name for name in _PREFERRED_KEX_CANDIDATES if name in supported]


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
    return f"{conn.username}@{conn.host}:{remote_path}"


def _fingerprint_from_key_data(key_data: str) -> str:
    try:
        key_bytes = base64.b64decode(key_data.encode("ascii"), validate=True)
    except Exception as exc:
        raise SshKnownHostsError("Invalid SSH public key payload in known_hosts/ssh-keyscan output.") from exc
    digest = hashlib.sha256(key_bytes).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _is_unsupported_kex_error(stderr: str | None) -> bool:
    text = (stderr or "").lower()
    return "choose_kex" in text and "unsupported kex method" in text


def _fetch_ssh_ed25519_host_key_with_ssh(*, host: str, port: int, timeout_seconds: float) -> KnownHostKey | None:
    kex_algorithms = preferred_kex_algorithms_for_local_ssh()
    if not kex_algorithms:
        return None

    fd, tmp_name = tempfile.mkstemp(prefix=".vast-bootstrap-", suffix=".known_hosts")
    os.close(fd)
    tmp_known_hosts = Path(tmp_name)
    try:
        try:
            completed = subprocess.run(
                [
                    "ssh",
                    "-p",
                    str(port),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    f"UserKnownHostsFile={tmp_known_hosts}",
                    "-o",
                    "NumberOfPasswordPrompts=0",
                    "-o",
                    "HostKeyAlgorithms=ssh-ed25519",
                    "-o",
                    f"KexAlgorithms={','.join(kex_algorithms)}",
                    f"root@{host}",
                    "exit",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise SshBinaryNotFoundError("OpenSSH binary not found in PATH: ssh") from exc
        except subprocess.TimeoutExpired as exc:
            raise SshConnectionError(f"SSH probe timed out after {timeout_seconds:.1f}s") from exc

        key = lookup_known_host_key(known_hosts_path=tmp_known_hosts, host=host, port=port)
        if key is not None:
            return key

        stderr = completed.stderr.strip()
        if stderr:
            raise SshConnectionError(stderr)
        return None
    finally:
        if tmp_known_hosts.exists():
            tmp_known_hosts.unlink()
