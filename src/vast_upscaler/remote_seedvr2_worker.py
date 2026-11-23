from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

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


class RemoteSeedvr2WorkerReportError(RemoteSeedvr2WorkerError):
    """Raised when SeedVR2 report is missing or invalid."""


@dataclass(frozen=True)
class Seedvr2RemoteRequest:
    session_id: str
    local_input_path: Path
    local_output_path: Path
    local_report_path: Path
    profile: str
    remote_root_base: str = "/workspace/seedvr2-sessions"
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


def execute_remote_seedvr2_worker(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    request: Seedvr2RemoteRequest,
) -> Seedvr2RemoteResult:
    paths = _build_remote_paths(request)
    _prepare_remote_dirs(transport=transport, connection=connection, paths=paths)

    try:
        remote_input_path = transport.upload_file(
            connection,
            local_path=request.local_input_path,
            remote_dir=paths.input_dir,
            timeout_seconds=request.upload_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
        )
    except TransferError as exc:
        raise RemoteSeedvr2WorkerTransferError(f"Input upload failed: {exc}") from exc

    command = _build_seedvr2_command(
        input_path=remote_input_path,
        profile=request.profile,
        output_root=paths.output_dir,
        work_root=paths.work_dir,
        cache_dir=paths.cache_dir,
    )
    try:
        process_result = transport.run_remote_command(
            connection,
            command=command,
            timeout_seconds=request.process_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=False,
            check=False,
        )
    except SshTransportError as exc:
        raise RemoteSeedvr2WorkerProcessError(f"Remote SeedVR2 execution failed before completion: {exc}") from exc

    if process_result.returncode != 0:
        raise RemoteSeedvr2WorkerProcessError(
            f"Remote SeedVR2 exited with code {process_result.returncode}. "
            f"stderr: {(process_result.stderr or '').strip()}"
        )

    discovered = _discover_seedvr2_outputs(transport=transport, connection=connection, output_root=paths.output_dir)
    remote_report_path = discovered["report"]
    remote_output_path = discovered["output"]

    try:
        transport.download_file(
            connection,
            remote_path=remote_report_path,
            local_path=request.local_report_path,
            timeout_seconds=request.download_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            overwrite=True,
        )
    except TransferError as exc:
        raise RemoteSeedvr2WorkerTransferError(f"Could not download SeedVR2 report: {exc}") from exc

    report_payload = _load_seedvr2_report(request.local_report_path)
    status = str(report_payload.get("status") or "").lower()
    if status != "completed":
        raise RemoteSeedvr2WorkerReportError(f"SeedVR2 report status is not completed: {status or 'missing'}")

    try:
        transport.download_file(
            connection,
            remote_path=remote_output_path,
            local_path=request.local_output_path,
            timeout_seconds=request.download_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            overwrite=True,
        )
    except TransferError as exc:
        raise RemoteSeedvr2WorkerTransferError(f"Could not download SeedVR2 output video: {exc}") from exc

    return Seedvr2RemoteResult(
        session_id=request.session_id,
        command=command,
        remote_paths=paths,
        process_returncode=process_result.returncode,
        process_stdout=process_result.stdout,
        process_stderr=process_result.stderr,
        report_payload=report_payload,
        local_output_path=request.local_output_path,
        local_report_path=request.local_report_path,
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
        "python",
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
    cache_dir = posixpath.join(root, "cache")
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
    cmd = "mkdir -p {input_dir} {output_dir} {work_dir} {cache_dir} {logs_dir}".format(
        input_dir=shlex.quote(paths.input_dir),
        output_dir=shlex.quote(paths.output_dir),
        work_dir=shlex.quote(paths.work_dir),
        cache_dir=shlex.quote(paths.cache_dir),
        logs_dir=shlex.quote(paths.logs_dir),
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
    script = (
        "import json; from pathlib import Path; "
        f"root = Path({output_root!r}); "
        "reports = sorted([p for p in root.rglob('prototype-report.json') if p.is_file()], key=lambda p: p.stat().st_mtime); "
        "assert reports, 'No prototype-report.json found under output root'; "
        "report_path = reports[-1]; "
        "payload = json.loads(report_path.read_text(encoding='utf-8')); "
        "output_path = payload.get('output_file'); "
        "assert isinstance(output_path, str) and output_path.strip(), 'SeedVR2 report missing output_file'; "
        "print(json.dumps({'report': str(report_path), 'output': output_path}))"
    )
    try:
        result = transport.run_remote_command(
            connection,
            command=["python", "-c", script],
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
    return {"report": report, "output": output}


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


def _sanitize_session_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise RemoteSeedvr2WorkerError("session_id is empty after sanitization.")
    return cleaned
