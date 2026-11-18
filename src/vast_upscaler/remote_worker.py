"""Phase 4D remote worker execution using SSH/SCP transport."""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vast_upscaler.transfer import (
    OpenSshTransport,
    RetryPolicy,
    SshConnectionError,
    SshConnectionInfo,
    SshTimeoutError,
    SshTransportError,
    TransferError,
)
from vast_upscaler.video import VideoInfo, VideoProbeError, inspect_video, run_ffprobe

WORKER_PROGRESS_EVENT_PREFIX = "VAST_PROGRESS_JSON "
ProgressCallback = Callable[[dict[str, object]], None]


class RemoteWorkerError(RuntimeError):
    """Base exception for remote worker execution failures."""


class RemoteWorkerTransferError(RemoteWorkerError):
    """Raised when upload/download fails."""


class RemoteWorkerProcessError(RemoteWorkerError):
    """Raised when remote worker process exits with an error."""


class RemoteWorkerReportError(RemoteWorkerError):
    """Raised when report is missing, invalid, or incoherent."""


class RemoteWorkerOutputError(RemoteWorkerError):
    """Raised when output video is missing or invalid."""


class RemoteWorkerAmbiguousStateError(RemoteWorkerError):
    """Raised when process interruption leaves remote state uncertain."""


@dataclass(frozen=True)
class RemoteWorkerRequest:
    session_id: str
    local_input_path: Path
    local_output_path: Path
    local_report_path: Path
    target_width: int
    target_height: int
    profile: str = "balanced"
    video_crf: int = 20
    models_dir: str = "/opt/models"
    min_free_gb: float = 2.0
    remote_root_base: str = "/workspace/upscale-sessions"
    upload_timeout_seconds: float = 600.0
    process_timeout_seconds: float = 14_400.0
    download_timeout_seconds: float = 1800.0
    ffprobe_bin: str = "ffprobe"


@dataclass(frozen=True)
class RemoteExecutionPaths:
    root_dir: str
    input_dir: str
    output_dir: str
    logs_dir: str
    remote_input_path: str
    remote_output_path: str
    remote_report_path: str


@dataclass(frozen=True)
class RemoteWorkerResult:
    session_id: str
    command: list[str]
    remote_paths: RemoteExecutionPaths
    process_returncode: int
    process_stdout: str
    process_stderr: str
    report_payload: dict
    frames_processed: int
    video_info: VideoInfo
    output_frame_count: int | None
    local_output_path: Path
    local_report_path: Path


def execute_remote_worker(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    request: RemoteWorkerRequest,
    progress_callback: ProgressCallback | None = None,
) -> RemoteWorkerResult:
    paths = _build_remote_paths(request)
    _prepare_remote_dirs(transport=transport, connection=connection, paths=paths)

    try:
        if progress_callback is not None:
            progress_callback({"kind": "transfer", "stage": "upload", "status": "started"})
        uploaded_input = transport.upload_file(
            connection,
            local_path=request.local_input_path,
            remote_dir=paths.input_dir,
            timeout_seconds=request.upload_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        if progress_callback is not None:
            progress_callback({"kind": "transfer", "stage": "upload", "status": "completed"})
    except TransferError as exc:
        raise RemoteWorkerTransferError(f"Input upload failed: {exc}") from exc

    command = _build_worker_command(
        input_path=uploaded_input,
        output_path=paths.remote_output_path,
        report_path=paths.remote_report_path,
        request=request,
    )

    try:
        def _on_stdout_line(line: str) -> None:
            event = _parse_worker_progress_event(line)
            if event is None:
                return
            if progress_callback is not None:
                progress_callback(event)

        process_result = transport.run_remote_command(
            connection,
            command=command,
            timeout_seconds=request.process_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=False,
            check=False,
            stdout_line_callback=_on_stdout_line,
        )
    except (SshTimeoutError, SshConnectionError) as exc:
        _raise_ambiguous_after_probe(
            transport=transport,
            connection=connection,
            paths=paths,
            original_error=exc,
        )

    report_payload = _download_and_parse_report(
        transport=transport,
        connection=connection,
        remote_report_path=paths.remote_report_path,
        local_report_path=request.local_report_path,
        timeout_seconds=request.download_timeout_seconds,
    )

    if process_result.returncode != 0:
        details = _report_failure_details(report_payload)
        raise RemoteWorkerProcessError(
            f"Remote worker returned non-zero exit code {process_result.returncode}. "
            f"{details} stderr: {(process_result.stderr or '').strip()}"
        )
    frames_processed = _validate_report(report_payload=report_payload, expected_output_path=paths.remote_output_path)

    if progress_callback is not None:
        progress_callback({"kind": "transfer", "stage": "download", "status": "started"})
    local_output_path = _download_output_video(
        transport=transport,
        connection=connection,
        remote_output_path=paths.remote_output_path,
        local_output_path=request.local_output_path,
        timeout_seconds=request.download_timeout_seconds,
    )
    if progress_callback is not None:
        progress_callback({"kind": "transfer", "stage": "download", "status": "completed"})

    video_info = _validate_local_output_video(
        output_path=local_output_path,
        target_width=request.target_width,
        target_height=request.target_height,
        ffprobe_bin=request.ffprobe_bin,
    )
    output_frame_count = _probe_optional_frame_count(local_output_path, ffprobe_bin=request.ffprobe_bin)
    if output_frame_count is not None and output_frame_count <= 0:
        raise RemoteWorkerOutputError("Output frame count is available but invalid (<= 0).")

    return RemoteWorkerResult(
        session_id=request.session_id,
        command=command,
        remote_paths=paths,
        process_returncode=process_result.returncode,
        process_stdout=process_result.stdout,
        process_stderr=process_result.stderr,
        report_payload=report_payload,
        frames_processed=frames_processed,
        video_info=video_info,
        output_frame_count=output_frame_count,
        local_output_path=local_output_path,
        local_report_path=request.local_report_path,
    )


def _parse_worker_progress_event(line: str) -> dict[str, object] | None:
    if not line.startswith(WORKER_PROGRESS_EVENT_PREFIX):
        return None
    payload = line[len(WORKER_PROGRESS_EVENT_PREFIX) :]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "kind" not in parsed:
        parsed["kind"] = "worker"
    return parsed


def recover_remote_results(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    request: RemoteWorkerRequest,
) -> RemoteWorkerResult:
    paths = _build_remote_paths(request)

    report_payload = _download_and_parse_report(
        transport=transport,
        connection=connection,
        remote_report_path=paths.remote_report_path,
        local_report_path=request.local_report_path,
        timeout_seconds=request.download_timeout_seconds,
    )
    frames_processed = _validate_report(report_payload=report_payload, expected_output_path=paths.remote_output_path)

    local_output_path = _download_output_video(
        transport=transport,
        connection=connection,
        remote_output_path=paths.remote_output_path,
        local_output_path=request.local_output_path,
        timeout_seconds=request.download_timeout_seconds,
    )

    video_info = _validate_local_output_video(
        output_path=local_output_path,
        target_width=request.target_width,
        target_height=request.target_height,
        ffprobe_bin=request.ffprobe_bin,
    )
    output_frame_count = _probe_optional_frame_count(local_output_path, ffprobe_bin=request.ffprobe_bin)

    return RemoteWorkerResult(
        session_id=request.session_id,
        command=["recovery-download-only"],
        remote_paths=paths,
        process_returncode=0,
        process_stdout="",
        process_stderr="",
        report_payload=report_payload,
        frames_processed=frames_processed,
        video_info=video_info,
        output_frame_count=output_frame_count,
        local_output_path=local_output_path,
        local_report_path=request.local_report_path,
    )


def _prepare_remote_dirs(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    paths: RemoteExecutionPaths,
) -> None:
    quoted_root = shlex.quote(paths.root_dir)
    quoted_input = shlex.quote(paths.input_dir)
    quoted_output = shlex.quote(paths.output_dir)
    quoted_logs = shlex.quote(paths.logs_dir)
    command = (
        f"mkdir -p {quoted_input} {quoted_output} {quoted_logs} && "
        f"test -d {quoted_root} -a -w {quoted_root} && "
        f"test -d {quoted_input} -a -w {quoted_input} && "
        f"test -d {quoted_output} -a -w {quoted_output} && "
        f"test -d {quoted_logs} -a -w {quoted_logs}"
    )
    try:
        transport.run_remote_command(
            connection,
            command=command,
            timeout_seconds=30.0,
            retry_policy=RetryPolicy(max_attempts=3, retry_delay_seconds=2.0),
            idempotent=True,
            check=True,
        )
    except SshTransportError as exc:
        raise RemoteWorkerError(f"Failed to prepare remote directories: {exc}") from exc


def _download_and_parse_report(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    remote_report_path: str,
    local_report_path: Path,
    timeout_seconds: float,
) -> dict:
    try:
        transport.download_file(
            connection,
            remote_path=remote_report_path,
            local_path=local_report_path,
            timeout_seconds=timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=2, retry_delay_seconds=2.0),
            overwrite=False,
        )
    except TransferError as exc:
        raise RemoteWorkerReportError(f"Could not download worker report: {exc}") from exc

    try:
        payload = json.loads(local_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RemoteWorkerReportError("Worker report is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RemoteWorkerReportError("Worker report root must be a JSON object.")
    return payload


def _validate_report(*, report_payload: dict, expected_output_path: str) -> int:
    status = str(report_payload.get("status") or "").lower()
    if status != "completed":
        raise RemoteWorkerReportError(
            f"Worker report status is not completed: {status or 'missing'}. {_report_failure_details(report_payload)}"
        )

    frames_processed = report_payload.get("frames_processed")
    if not isinstance(frames_processed, int) or frames_processed <= 0:
        raise RemoteWorkerReportError("Worker report frames_processed must be a positive integer.")

    output_path = report_payload.get("output")
    if not isinstance(output_path, str) or not output_path.strip():
        raise RemoteWorkerReportError("Worker report output path is missing or invalid.")

    if posixpath.normpath(output_path) != posixpath.normpath(expected_output_path):
        raise RemoteWorkerReportError(
            "Worker report output path does not match the expected remote output path."
        )

    return frames_processed


def _report_failure_details(report_payload: dict) -> str:
    notes = report_payload.get("notes")
    if isinstance(notes, list):
        text_notes = [str(item).strip() for item in notes if str(item).strip()]
        if text_notes:
            preview = "; ".join(text_notes[:3])
            return f"Report notes: {preview}."
    message = report_payload.get("message")
    if isinstance(message, str) and message.strip():
        return f"Report message: {message.strip()}."
    return "No detailed worker failure note available."


def _download_output_video(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    remote_output_path: str,
    local_output_path: Path,
    timeout_seconds: float,
) -> Path:
    try:
        return transport.download_file(
            connection,
            remote_path=remote_output_path,
            local_path=local_output_path,
            timeout_seconds=timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=2, retry_delay_seconds=2.0),
            overwrite=False,
        )
    except TransferError as exc:
        raise RemoteWorkerOutputError(f"Could not download output video: {exc}") from exc


def _validate_local_output_video(
    *,
    output_path: Path,
    target_width: int,
    target_height: int,
    ffprobe_bin: str,
) -> VideoInfo:
    try:
        info = inspect_video(output_path, ffprobe_bin=ffprobe_bin)
    except VideoProbeError as exc:
        raise RemoteWorkerOutputError(f"Output video ffprobe validation failed: {exc}") from exc

    if info.width != target_width or info.height != target_height:
        raise RemoteWorkerOutputError(
            f"Output resolution mismatch: expected {target_width}x{target_height}, got {info.width}x{info.height}"
        )

    if info.duration_seconds is None or info.duration_seconds <= 0:
        raise RemoteWorkerOutputError("Output duration is missing or invalid.")

    return info


def _probe_optional_frame_count(video_path: Path, *, ffprobe_bin: str) -> int | None:
    payload = run_ffprobe(video_path=video_path, ffprobe_bin=ffprobe_bin)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return None
    video_stream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        return None

    for key in ("nb_read_frames", "nb_frames"):
        value = video_stream.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _raise_ambiguous_after_probe(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    paths: RemoteExecutionPaths,
    original_error: Exception,
) -> None:
    probe_cmd = (
        f"test -f {shlex.quote(paths.remote_report_path)} && echo REPORT_PRESENT=1 || echo REPORT_PRESENT=0; "
        f"test -f {shlex.quote(paths.remote_output_path)} && echo OUTPUT_PRESENT=1 || echo OUTPUT_PRESENT=0"
    )

    try:
        probe = transport.run_remote_command(
            connection,
            command=probe_cmd,
            timeout_seconds=30.0,
            retry_policy=RetryPolicy(max_attempts=1),
            idempotent=True,
            check=False,
        )
    except SshTransportError as probe_exc:
        raise RemoteWorkerAmbiguousStateError(
            "Worker execution was interrupted and remote state could not be verified. "
            "Do not retry processing automatically; inspect remote session paths first."
        ) from probe_exc

    report_present = "REPORT_PRESENT=1" in (probe.stdout or "")
    output_present = "OUTPUT_PRESENT=1" in (probe.stdout or "")
    raise RemoteWorkerAmbiguousStateError(
        "Worker execution was interrupted. "
        f"Remote probe: report_present={report_present}, output_present={output_present}. "
        "Do not restart processing automatically before reconciliation."
    ) from original_error


def _build_worker_command(
    *,
    input_path: str,
    output_path: str,
    report_path: str,
    request: RemoteWorkerRequest,
) -> list[str]:
    return [
        "python",
        "/app/worker/process.py",
        "--input",
        input_path,
        "--output",
        output_path,
        "--target-width",
        str(request.target_width),
        "--target-height",
        str(request.target_height),
        "--profile",
        request.profile,
        "--video-crf",
        str(request.video_crf),
        "--models-dir",
        request.models_dir,
        "--report",
        report_path,
        "--min-free-gb",
        str(request.min_free_gb),
    ]


def _build_remote_paths(request: RemoteWorkerRequest) -> RemoteExecutionPaths:
    session_key = _sanitize_session_id(request.session_id)
    root = posixpath.join(request.remote_root_base.rstrip("/"), session_key)
    input_dir = posixpath.join(root, "input")
    output_dir = posixpath.join(root, "output")
    logs_dir = posixpath.join(root, "logs")

    input_name = request.local_input_path.name
    output_name = request.local_output_path.name
    if not output_name:
        raise RemoteWorkerError("local_output_path must reference a file name.")

    return RemoteExecutionPaths(
        root_dir=root,
        input_dir=input_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        remote_input_path=posixpath.join(input_dir, input_name),
        remote_output_path=posixpath.join(output_dir, output_name),
        remote_report_path=posixpath.join(logs_dir, "report.json"),
    )


def _sanitize_session_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise RemoteWorkerError("session_id is empty after sanitization.")
    return cleaned
