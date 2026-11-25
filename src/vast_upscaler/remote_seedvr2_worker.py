from __future__ import annotations

import json
import posixpath
import re
import shlex
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from vast_upscaler.transfer import (
    OpenSshTransport,
    RetryPolicy,
    SshConnectionInfo,
    SshTransportError,
    TransferError,
)


class RemoteSeedvr2WorkerError(RuntimeError):
    """Base exception for SeedVR2 remote execution failures."""


class RemoteSeedvr2WorkerTransferError(RemoteSeedvr2WorkerError):
    """Raised when upload/download fails."""


class RemoteSeedvr2WorkerProcessError(RemoteSeedvr2WorkerError):
    """Raised when remote SeedVR2 process exits with an error."""


class RemoteSeedvr2WorkerPreparationError(RemoteSeedvr2WorkerError):
    """Raised when remote preparation fails before SeedVR2 inference starts."""


class RemoteSeedvr2WorkerReportError(RemoteSeedvr2WorkerError):
    """Raised when SeedVR2 report is missing or invalid."""


class RemoteSeedvr2WorkerRecoveryError(RemoteSeedvr2WorkerError):
    """Raised when inference completed remotely but artifact recovery failed."""


ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class Seedvr2RemoteRequest:
    session_id: str
    local_input_path: Path
    local_output_path: Path
    local_report_path: Path
    profile: str
    remote_root_base: str = "/workspace/seedvr2-sessions"
    shared_cache_dir: str = "/workspace/seedvr2-cache"
    upload_timeout_seconds: float = 600.0
    process_timeout_seconds: float = 28_800.0
    download_timeout_seconds: float = 1800.0


@dataclass(frozen=True)
class Seedvr2RemotePaths:
    root_dir: str
    input_dir: str
    output_dir: str
    work_dir: str
    cache_dir: str
    logs_dir: str
    remote_input_path: str


@dataclass(frozen=True)
class Seedvr2RemoteResult:
    session_id: str
    command: list[str]
    remote_paths: Seedvr2RemotePaths
    process_returncode: int
    process_stdout: str
    process_stderr: str
    report_payload: dict
    local_output_path: Path
    local_report_path: Path
    local_log_path: Path | None


_SEEDVR2_PYTHON = "/opt/seedvr2-venv/bin/python"
_CHECKPOINT_LOCKFILE = ".seedvr2-checkpoints.lock"
_CHECKPOINT_LOCK_TIMEOUT_SECONDS = 900.0
_CHECKPOINT_RETRY_DELAY_SECONDS = 0.2
_CHECKPOINT_METADATA = {
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": {
        "repo": "numz/SeedVR2_comfyUI",
        "sha256": "3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff",
    },
    "ema_vae_fp16.safetensors": {
        "repo": "numz/SeedVR2_comfyUI",
        "sha256": "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
    },
}


def execute_remote_seedvr2_worker(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    request: Seedvr2RemoteRequest,
    progress_callback: ProgressCallback | None = None,
) -> Seedvr2RemoteResult:
    paths = _build_remote_paths(request)
    _prepare_remote_dirs(transport=transport, connection=connection, paths=paths)

    try:
        if progress_callback is not None:
            progress_callback({"kind": "transfer", "stage": "upload", "status": "started"})
        remote_input_path = transport.upload_file(
            connection,
            local_path=request.local_input_path,
            remote_dir=paths.input_dir,
            timeout_seconds=request.upload_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
        )
    except TransferError as exc:
        raise RemoteSeedvr2WorkerTransferError(f"Input upload failed: {exc}") from exc
    if progress_callback is not None:
        progress_callback({"kind": "transfer", "stage": "upload", "status": "completed"})

    _ensure_shared_checkpoints(
        transport=transport,
        connection=connection,
        cache_dir=paths.cache_dir,
        progress_callback=progress_callback,
    )

    command = _build_seedvr2_command(
        input_path=remote_input_path,
        profile=request.profile,
        output_root=paths.output_dir,
        work_root=paths.work_dir,
        cache_dir=paths.cache_dir,
    )
    process_result = None
    try:
        if progress_callback is not None:
            progress_callback({"kind": "worker", "stage": "seedvr2-runner", "status": "started", "determinate": False})
        progress_state = _Seedvr2ProgressState()

        def _on_stdout_line(line: str) -> None:
            event = _parse_seedvr2_progress_line(line=line, state=progress_state)
            if event is not None and progress_callback is not None:
                progress_callback(event)
            if progress_callback is not None and _should_emit_log_line(line=line, state=progress_state):
                progress_callback({"kind": "worker-log", "line": line[-800:]})

        shell_command = _as_shell_command_with_stderr_merge(command)
        process_result = transport.run_remote_command(
            connection,
            command=shell_command,
            timeout_seconds=request.process_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=False,
            check=False,
            stdout_line_callback=_on_stdout_line,
        )
    except SshTransportError as exc:
        raise RemoteSeedvr2WorkerProcessError(f"Remote SeedVR2 execution failed before completion: {exc}") from exc

    if process_result.returncode != 0:
        raise RemoteSeedvr2WorkerProcessError(
            f"Remote SeedVR2 exited with code {process_result.returncode}. "
            f"stderr: {(process_result.stderr or '').strip()}"
        )

    return _download_seedvr2_artifacts(
        transport=transport,
        connection=connection,
        request=request,
        paths=paths,
        command=command,
        process_returncode=process_result.returncode,
        process_stdout=process_result.stdout,
        process_stderr=process_result.stderr,
        progress_callback=progress_callback,
    )


def recover_remote_seedvr2_results(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    request: Seedvr2RemoteRequest,
    progress_callback: ProgressCallback | None = None,
) -> Seedvr2RemoteResult:
    paths = _build_remote_paths(request)
    return _download_seedvr2_artifacts(
        transport=transport,
        connection=connection,
        request=request,
        paths=paths,
        command=[],
        process_returncode=0,
        process_stdout="",
        process_stderr="",
        progress_callback=progress_callback,
    )


def _build_seedvr2_command(
    *,
    input_path: str,
    profile: str,
    output_root: str,
    work_root: str,
    cache_dir: str,
) -> list[str]:
    return [
        _SEEDVR2_PYTHON,
        "/opt/prototype/seedvr2_prototype.py",
        "--input-video",
        input_path,
        "--profile",
        profile,
        "--output-root",
        output_root,
        "--work-root",
        work_root,
        "--cache-dir",
        cache_dir,
        "--input-is-excerpt",
    ]


def _build_remote_paths(request: Seedvr2RemoteRequest) -> Seedvr2RemotePaths:
    session_key = _sanitize_session_id(request.session_id)
    root = posixpath.join(request.remote_root_base.rstrip("/"), session_key)
    input_dir = posixpath.join(root, "input")
    output_dir = posixpath.join(root, "output")
    work_dir = posixpath.join(root, "work")
    cache_dir = posixpath.normpath(request.shared_cache_dir)
    logs_dir = posixpath.join(root, "logs")

    input_name = request.local_input_path.name
    if not input_name:
        raise RemoteSeedvr2WorkerError("local_input_path must reference a file name.")

    return Seedvr2RemotePaths(
        root_dir=root,
        input_dir=input_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        remote_input_path=posixpath.join(input_dir, input_name),
    )


def _prepare_remote_dirs(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    paths: Seedvr2RemotePaths,
) -> None:
    cmd = (
        f"mkdir -p {shlex.quote(paths.input_dir)} {shlex.quote(paths.output_dir)} "
        f"{shlex.quote(paths.work_dir)} {shlex.quote(paths.cache_dir)} {shlex.quote(paths.logs_dir)}"
    )
    try:
        transport.run_remote_command(
            connection,
            command=cmd,
            timeout_seconds=30.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            check=True,
        )
    except SshTransportError as exc:
        raise RemoteSeedvr2WorkerError(f"Could not prepare SeedVR2 remote directories: {exc}") from exc


def _discover_seedvr2_outputs(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    output_root: str,
) -> dict[str, str]:
    script = _build_seedvr2_discovery_script(output_root=output_root)
    try:
        result = transport.run_remote_command(
            connection,
            command=[_SEEDVR2_PYTHON, "-c", script],
            timeout_seconds=30.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            check=True,
        )
    except SshTransportError as exc:
        raise RemoteSeedvr2WorkerReportError(f"Could not discover SeedVR2 report/output paths: {exc}") from exc

    raw = (result.stdout or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteSeedvr2WorkerReportError("SeedVR2 discovery output is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RemoteSeedvr2WorkerReportError("SeedVR2 discovery output must be a JSON object.")
    report = payload.get("report")
    output = payload.get("output")
    if not isinstance(report, str) or not report.strip():
        raise RemoteSeedvr2WorkerReportError("SeedVR2 discovery result missing report path.")
    if not isinstance(output, str) or not output.strip():
        raise RemoteSeedvr2WorkerReportError("SeedVR2 discovery result missing output path.")
    log_path = payload.get("log")
    if log_path is not None and (not isinstance(log_path, str) or not log_path.strip()):
        raise RemoteSeedvr2WorkerReportError("SeedVR2 discovery result contains invalid log path.")
    result = {"report": report, "output": output}
    if isinstance(log_path, str):
        result["log"] = log_path
    return result


def _download_seedvr2_artifacts(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    request: Seedvr2RemoteRequest,
    paths: Seedvr2RemotePaths,
    command: list[str],
    process_returncode: int,
    process_stdout: str,
    process_stderr: str,
    progress_callback: ProgressCallback | None,
) -> Seedvr2RemoteResult:
    tmp_suffix = f".{uuid4().hex[:8]}.part"
    report_tmp = request.local_report_path.with_name(request.local_report_path.name + tmp_suffix)
    output_tmp = request.local_output_path.with_name(request.local_output_path.name + tmp_suffix)
    log_tmp: Path | None = None

    def _cleanup_tmp_files() -> None:
        for candidate in (report_tmp, output_tmp, log_tmp):
            if candidate is not None and candidate.exists():
                candidate.unlink(missing_ok=True)

    def _existing_identity(report_path: Path) -> tuple[str | None, str | None]:
        if not report_path.exists() or not report_path.is_file():
            return (None, None)
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return (None, None)
        if not isinstance(payload, dict):
            return (None, None)
        engine = payload.get("engine")
        job_id = payload.get("job_id")
        return (
            str(engine) if isinstance(engine, str) and engine.strip() else None,
            str(job_id) if isinstance(job_id, str) and job_id.strip() else None,
        )

    def _guard_overwrite_of_other_execution() -> None:
        if not request.local_report_path.exists() and not request.local_output_path.exists():
            return
        engine, job_id = _existing_identity(request.local_report_path)
        if engine == "seedvr2" and job_id == request.session_id:
            return
        raise RemoteSeedvr2WorkerRecoveryError("Refusing to overwrite existing local artifacts from another execution.")

    def _publish_atomically(*, local_log_path: Path | None, staged_log_path: Path | None) -> None:
        rollback_suffix = f".rollback.{uuid4().hex[:8]}"
        backups: dict[str, Path] = {}
        published_paths: list[Path] = []
        failed_restore_backups: set[Path] = set()

        def _backup_if_exists(label: str, final_path: Path) -> None:
            if final_path.exists():
                backup_path = final_path.with_name(final_path.name + rollback_suffix)
                final_path.replace(backup_path)
                backups[label] = backup_path

        def _target_for_label(label: str) -> Path | None:
            if label == "output":
                return request.local_output_path
            if label == "report":
                return request.local_report_path
            if label == "log":
                return local_log_path
            return None

        request.local_output_path.parent.mkdir(parents=True, exist_ok=True)
        request.local_report_path.parent.mkdir(parents=True, exist_ok=True)
        if local_log_path is not None:
            local_log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            _backup_if_exists("output", request.local_output_path)
            _backup_if_exists("report", request.local_report_path)
            if local_log_path is not None:
                _backup_if_exists("log", local_log_path)

            output_tmp.replace(request.local_output_path)
            published_paths.append(request.local_output_path)
            report_tmp.replace(request.local_report_path)
            published_paths.append(request.local_report_path)
            if local_log_path is not None and staged_log_path is not None and staged_log_path.exists():
                staged_log_path.replace(local_log_path)
                published_paths.append(local_log_path)
        except OSError as exc:
            for published in reversed(published_paths):
                if published.exists():
                    published.unlink(missing_ok=True)
            for label, backup in backups.items():
                target = _target_for_label(label)
                if target is None:
                    continue
                try:
                    backup.replace(target)
                except OSError:
                    failed_restore_backups.add(backup)
            raise RemoteSeedvr2WorkerRecoveryError(f"Could not publish SeedVR2 local artifacts: {exc}") from exc
        finally:
            for backup in backups.values():
                if backup in failed_restore_backups:
                    continue
                if backup.exists():
                    backup.unlink(missing_ok=True)

    _guard_overwrite_of_other_execution()
    try:
        discovered = _discover_seedvr2_outputs(transport=transport, connection=connection, output_root=paths.output_dir)
    except RemoteSeedvr2WorkerReportError as exc:
        _cleanup_tmp_files()
        raise RemoteSeedvr2WorkerRecoveryError(f"Could not discover SeedVR2 output artifacts: {exc}") from exc
    remote_report_path = discovered["report"]
    remote_output_path = discovered["output"]
    remote_log_path = discovered.get("log")

    try:
        if progress_callback is not None:
            progress_callback({"kind": "transfer", "stage": "download", "status": "started"})
        transport.download_file(
            connection,
            remote_path=remote_report_path,
            local_path=report_tmp,
            timeout_seconds=request.download_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            overwrite=True,
        )
    except TransferError as exc:
        _cleanup_tmp_files()
        raise RemoteSeedvr2WorkerRecoveryError(f"Could not download SeedVR2 report: {exc}") from exc

    try:
        report_payload = _load_seedvr2_report(report_tmp)
    except RemoteSeedvr2WorkerReportError as exc:
        _cleanup_tmp_files()
        raise RemoteSeedvr2WorkerRecoveryError(f"SeedVR2 report is missing or invalid after inference: {exc}") from exc
    status = str(report_payload.get("status") or "").lower()
    if status != "completed":
        if status in {"failed", "error", "cancelled"}:
            _cleanup_tmp_files()
            raise RemoteSeedvr2WorkerProcessError(f"SeedVR2 inference report status: {status}")
        _cleanup_tmp_files()
        raise RemoteSeedvr2WorkerRecoveryError(f"SeedVR2 report status is not completed: {status or 'missing'}")

    try:
        _validate_remote_path_within_root(path=remote_output_path, root=paths.output_dir, label="output")
        if remote_log_path is not None:
            _validate_remote_path_within_root(path=remote_log_path, root=paths.output_dir, label="log")
    except RemoteSeedvr2WorkerReportError as exc:
        _cleanup_tmp_files()
        raise RemoteSeedvr2WorkerRecoveryError(f"Discovered SeedVR2 artifact path is invalid: {exc}") from exc

    try:
        transport.download_file(
            connection,
            remote_path=remote_output_path,
            local_path=output_tmp,
            timeout_seconds=request.download_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            overwrite=True,
        )
    except TransferError as exc:
        _cleanup_tmp_files()
        raise RemoteSeedvr2WorkerRecoveryError(f"Could not download SeedVR2 output video: {exc}") from exc

    local_log_path: Path | None = None
    if remote_log_path is not None:
        local_log_path = request.local_report_path.with_suffix(request.local_report_path.suffix + ".run.log")
        log_tmp = local_log_path.with_name(local_log_path.name + tmp_suffix)
        try:
            transport.download_file(
                connection,
                remote_path=remote_log_path,
                local_path=log_tmp,
                timeout_seconds=request.download_timeout_seconds,
                retry_policy=RetryPolicy(max_attempts=1),
                overwrite=True,
            )
        except TransferError:
            if log_tmp.exists():
                log_tmp.unlink(missing_ok=True)
            local_log_path = None
            log_tmp = None

    report_payload["engine"] = "seedvr2"
    report_payload["job_id"] = request.session_id
    report_payload["artifact_paths"] = {
        "output": str(request.local_output_path),
        "report": str(request.local_report_path),
        "log": str(local_log_path) if local_log_path is not None else None,
    }
    try:
        report_tmp.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _publish_atomically(local_log_path=local_log_path, staged_log_path=log_tmp)
    except (OSError, RemoteSeedvr2WorkerRecoveryError) as exc:
        _cleanup_tmp_files()
        if isinstance(exc, RemoteSeedvr2WorkerRecoveryError):
            raise
        raise RemoteSeedvr2WorkerRecoveryError(f"Could not finalize SeedVR2 local artifacts: {exc}") from exc

    if progress_callback is not None:
        progress_callback({"kind": "transfer", "stage": "download", "status": "completed"})

    return Seedvr2RemoteResult(
        session_id=request.session_id,
        command=command,
        remote_paths=paths,
        process_returncode=process_returncode,
        process_stdout=process_stdout,
        process_stderr=process_stderr,
        report_payload=report_payload,
        local_output_path=request.local_output_path,
        local_report_path=request.local_report_path,
        local_log_path=local_log_path,
    )


def _load_seedvr2_report(report_path: Path) -> dict:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RemoteSeedvr2WorkerReportError(f"Could not read SeedVR2 report: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RemoteSeedvr2WorkerReportError("SeedVR2 report is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RemoteSeedvr2WorkerReportError("SeedVR2 report root must be a JSON object.")
    return payload


def _build_seedvr2_discovery_script(*, output_root: str) -> str:
    return textwrap.dedent(
        f"""
        import json
        from pathlib import Path

        root = Path({output_root!r}).resolve()
        reports = sorted(
            [p for p in root.rglob("prototype-report.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
        )
        if not reports:
            raise RuntimeError("No prototype-report.json found under output root")

        report_path = reports[-1].resolve(strict=True)
        payload = json.loads(report_path.read_text(encoding="utf-8"))

        output_path = payload.get("output_file")
        if not isinstance(output_path, str) or not output_path.strip():
            raise RuntimeError("SeedVR2 report missing output_file")

        output_resolved = Path(output_path).resolve(strict=True)
        if not output_resolved.is_file():
            raise RuntimeError("SeedVR2 output_file does not exist")

        log_path = payload.get("log_file")
        log_resolved = None
        if isinstance(log_path, str) and log_path.strip():
            candidate = Path(log_path).resolve(strict=False)
            if candidate.exists() and candidate.is_file():
                log_resolved = candidate.resolve(strict=True)

        for label, candidate in (("report", report_path), ("output", output_resolved)):
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"SeedVR2 {{label}} path is outside output root: {{candidate}}") from exc
        if log_resolved is not None:
            try:
                log_resolved.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"SeedVR2 log path is outside output root: {{log_resolved}}") from exc

        print(
            json.dumps(
                {{
                    "report": str(report_path),
                    "output": str(output_resolved),
                    "log": str(log_resolved) if log_resolved is not None else None,
                }}
            )
        )
        """
    ).strip()


def _sanitize_session_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise RemoteSeedvr2WorkerError("session_id is empty after sanitization.")
    return cleaned


def _validate_remote_path_within_root(*, path: str, root: str, label: str) -> None:
    normalized_path = posixpath.normpath(path)
    normalized_root = posixpath.normpath(root)
    if not normalized_path.startswith("/"):
        raise RemoteSeedvr2WorkerReportError(f"SeedVR2 {label} path is not absolute: {path}")
    try:
        common_root = posixpath.commonpath([normalized_path, normalized_root])
    except ValueError as exc:
        raise RemoteSeedvr2WorkerReportError(f"SeedVR2 {label} path validation failed: {path}") from exc
    if common_root != normalized_root:
        raise RemoteSeedvr2WorkerReportError(
            f"SeedVR2 {label} path is outside output root: {path} (root: {normalized_root})"
        )


def _as_shell_command_with_stderr_merge(command: list[str]) -> str:
    return f"{' '.join(shlex.quote(item) for item in command)} 2>&1"


def _ensure_shared_checkpoints(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    cache_dir: str,
    progress_callback: ProgressCallback | None,
) -> None:
    if progress_callback is not None:
        progress_callback({"kind": "worker", "stage": "checkpoint-cache", "status": "started", "determinate": False})
    script = _build_checkpoint_cache_script(cache_dir=cache_dir)
    try:
        transport.run_remote_command(
            connection,
            command=[_SEEDVR2_PYTHON, "-c", script],
            timeout_seconds=_CHECKPOINT_LOCK_TIMEOUT_SECONDS,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            check=True,
        )
    except SshTransportError as exc:
        raise RemoteSeedvr2WorkerPreparationError(f"SeedVR2 checkpoint cache preparation failed: {exc}") from exc
    if progress_callback is not None:
        progress_callback({"kind": "worker", "stage": "checkpoint-cache", "status": "completed", "determinate": False})


def _build_checkpoint_cache_script(
    *,
    cache_dir: str,
    metadata_override: dict[str, dict[str, str]] | None = None,
    lock_timeout_seconds: float = _CHECKPOINT_LOCK_TIMEOUT_SECONDS,
    lock_retry_delay_seconds: float = _CHECKPOINT_RETRY_DELAY_SECONDS,
    hold_lock_seconds: float = 0.0,
) -> str:
    metadata = {
        filename: {
            "repo": values["repo"],
            "sha256": values["sha256"],
            "url": values.get("url", f"https://huggingface.co/{values['repo']}/resolve/main/{filename}"),
        }
        for filename, values in (metadata_override or _CHECKPOINT_METADATA).items()
    }
    return textwrap.dedent(
        f"""
        import errno
        import fcntl
        import hashlib
        import json
        import time
        import urllib.request
        from pathlib import Path

        cache = Path({cache_dir!r})
        cache.mkdir(parents=True, exist_ok=True)
        lock_path = cache / {_CHECKPOINT_LOCKFILE!r}
        started = time.time()
        lock_handle = lock_path.open("a+b")
        lock_acquired = False
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.time() - started > {lock_timeout_seconds!r}:
                    raise RuntimeError("Timeout while waiting for SeedVR2 checkpoint lock")
                time.sleep({lock_retry_delay_seconds!r})

        try:
            if {hold_lock_seconds!r} > 0:
                time.sleep({hold_lock_seconds!r})
            meta = {json.dumps(metadata, separators=(",", ":"))}

            def compute_sha256(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as fh:
                    while True:
                        chunk = fh.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                return digest.hexdigest().lower()

            for filename, item in meta.items():
                target = cache / filename
                expected_sha = item["sha256"].lower()
                if target.exists() and target.is_file():
                    digest = compute_sha256(target)
                    if digest == expected_sha:
                        continue
                    target.unlink()

                tmp = cache / (filename + ".part")
                if tmp.exists():
                    tmp.unlink()

                try:
                    with urllib.request.urlopen(item["url"], timeout=120) as response, tmp.open("wb") as fh:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            fh.write(chunk)
                except Exception:
                    if tmp.exists():
                        tmp.unlink()
                    raise

                digest = compute_sha256(tmp)
                if digest != expected_sha:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"Checkpoint checksum mismatch for {{filename}}")
                tmp.replace(target)
        finally:
            if lock_acquired:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        """
    ).strip()


class _Seedvr2ProgressState:
    def __init__(self) -> None:
        self.last_progress: float = 0.0
        self.last_log_emit_line: int = 0
        self.seen_lines: int = 0
        self.current_stage: str | None = None
        self.chunk_total: int | None = None
        self.completed_chunks: int = 0


def _parse_seedvr2_progress_line(*, line: str, state: _Seedvr2ProgressState) -> dict[str, object] | None:
    state.seen_lines += 1
    lowered = line.lower()

    stage = None
    if "vae" in lowered and "encode" in lowered:
        stage = "vae-encode"
    elif "dit" in lowered or "upscal" in lowered:
        stage = "dit-upscale"
    elif "vae" in lowered and "decode" in lowered:
        stage = "vae-decode"
    elif "mux" in lowered or "ffmpeg" in lowered:
        stage = "final-encode"

    if stage and stage != state.current_stage:
        state.current_stage = stage
        return {"kind": "worker", "stage": stage, "determinate": False}

    chunk_match = re.search(r"chunk\s*(\d+)\s*/\s*(\d+)", lowered)
    if chunk_match:
        current_chunk = int(chunk_match.group(1))
        chunk_total = int(chunk_match.group(2))
        if chunk_total > 0:
            state.chunk_total = chunk_total
            completed_chunks = max(0, min(chunk_total, current_chunk - 1))
            state.completed_chunks = max(state.completed_chunks, completed_chunks)
            progress = max(state.last_progress, min(0.999, state.completed_chunks / chunk_total))
            state.last_progress = progress
        else:
            progress = state.last_progress
        return {
            "kind": "worker",
            "stage": state.current_stage or "dit-upscale",
            "chunk_current": current_chunk,
            "chunk_total": chunk_total,
            "chunks_completed": state.completed_chunks,
            "progress": progress,
            "determinate": False,
        }

    frame_match = re.search(r"(?:frame|frames|image|images)\s*[:=]?\s*(\d+)(?:\s*/\s*(\d+))?", lowered)
    if frame_match:
        done = int(frame_match.group(1))
        total = frame_match.group(2)
        payload: dict[str, object] = {"kind": "worker", "stage": state.current_stage or "dit-upscale", "frames_done": done}
        if total is not None:
            total_int = int(total)
            if total_int > 0:
                progress = max(state.last_progress, min(1.0, done / total_int))
                state.last_progress = progress
                payload["frames_total"] = total_int
                payload["progress"] = progress
        return payload

    return None


def _should_emit_log_line(*, line: str, state: _Seedvr2ProgressState) -> bool:
    if not line.strip():
        return False
    important_tokens = ("error", "warning", "traceback", "oom", "cuda", "failed")
    lowered = line.lower()
    if any(token in lowered for token in important_tokens):
        state.last_log_emit_line = state.seen_lines
        return True
    if state.seen_lines - state.last_log_emit_line >= 25:
        state.last_log_emit_line = state.seen_lines
        return True
    return False
