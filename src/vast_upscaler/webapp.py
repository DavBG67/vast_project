from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from vast_upscaler.config import Settings, load_config
from vast_upscaler.orchestrator import (
    OrchestrationError,
    OrchestrationOptions,
    SessionExecutionLock,
    _build_trusted_connection,
    _resolve_private_key,
    _wait_for_ssh_tcp_availability,
    run_upscale_workflow,
)
from vast_upscaler.preprocessing_planner import (
    PlannerUserChoices,
    PreprocessingPlannerError,
    build_preprocessing_plan,
    load_analysis_payload_from_file,
)
from vast_upscaler.provisioning_history import ProvisioningAttemptTracker
from vast_upscaler.remote_worker import (
    RemoteWorkerRequest,
    execute_remote_worker,
)
from vast_upscaler.session_store import SessionStore
from vast_upscaler.source_analysis import (
    SourceAnalysisError,
    VisualDiagnosticsOptions,
    analyze_source_video,
    analyze_source_video_report,
)
from vast_upscaler.transfer import OpenSshTransport
from vast_upscaler.vast_client import VastClient, VastClientConfig
from vast_upscaler.vast_write_client import VastWriteClient

RunMode = Literal["automatic", "manual"]
SortBy = Literal["price", "reliability", "performance"]
SshBootstrapMode = Literal["tofu", "strict"]

_DEFAULT_WORKER_IMAGE = ""
_WORKER_IMAGE_HISTORY_PATH = Path.home() / ".vast_upscaler" / "web_worker_images.json"
_WORKER_IMAGE_PLACEHOLDERS = {
    "ghcr.io/acme/worker:test",
    "ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<sha>",
    "<image>",
    "example/image:tag",
}


class WebWorkflowCancelled(RuntimeError):
    pass


class StartJobRequest(BaseModel):
    mode: RunMode
    input_path: str
    output_path: str | None = None
    report_path: str | None = None
    target_width: int | None = Field(default=None, ge=1)
    target_height: int | None = Field(default=None, ge=1)
    gpu_model: str | None = None
    worker_image: str | None = None
    sort_by: SortBy = "price"
    offer_id: int | None = None
    processing_fps: float | None = Field(default=None, gt=0)
    instance_id: int | None = None
    ssh_bootstrap_mode: SshBootstrapMode = "tofu"


class PromptReplyRequest(BaseModel):
    approved: bool | None = None
    fingerprint: str | None = None


class SourceAnalysisRequest(BaseModel):
    input_path: str
    include_visual_diagnostics: bool = False
    diagnostic_options: VisualDiagnosticsOptions | None = None


class PreprocessingPlanRequest(BaseModel):
    input_path: str | None = None
    analysis_payload: dict[str, Any] | None = None
    analysis_report_path: str | None = None
    include_visual_diagnostics: bool = False
    diagnostic_options: VisualDiagnosticsOptions | None = None
    user_choices: PlannerUserChoices | None = None


@dataclass(frozen=True)
class WorkflowExecutionResult:
    mode: RunMode
    session_id: str | None
    state: str
    output_path: str | None
    report_path: str | None
    duration_seconds: float
    estimated_total_usd: float | None
    instance_id: int | None
    instance_state: str | None
    cleanup_verified: bool
    cleanup_warning: str | None


@dataclass(frozen=True)
class RunCallbacks:
    report_provisioning_status: Callable[[dict[str, str]], None]
    report_worker_progress: Callable[[dict[str, object]], None]
    confirm_paid: Callable[[str], bool]
    confirm_ssh_trust: Callable[[str], bool]
    request_verified_fingerprint: Callable[[str], str | None]
    raise_if_cancelled: Callable[[], None]


class _Job:
    def __init__(self, *, request: StartJobRequest, defaults: dict[str, Any]) -> None:
        self.job_id = f"job-{uuid4().hex[:12]}"
        self.request = request
        self.defaults = defaults
        self.created_at = datetime.now(UTC).isoformat()
        self.updated_at = self.created_at
        self.status = "pending"
        self.error: str | None = None
        self.result: WorkflowExecutionResult | None = None
        self.events: list[dict[str, Any]] = []
        self.next_event_id = 1
        self.condition = threading.Condition()
        self.thread: threading.Thread | None = None
        self.cancel_requested = False
        self.pending_prompt: dict[str, Any] | None = None
        self.answered_prompt_ids: set[str] = set()

    def is_active(self) -> bool:
        return self.status in {"pending", "running"}

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self.condition:
            event = {
                "id": self.next_event_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "type": event_type,
                "payload": payload,
            }
            self.next_event_id += 1
            self.events.append(event)
            self.updated_at = event["timestamp"]
            self.condition.notify_all()


class WebTaskManager:
    def __init__(
        self,
        *,
        config_path: Path,
        auto_runner: Callable[[StartJobRequest, Settings, RunCallbacks], WorkflowExecutionResult],
        manual_runner: Callable[[StartJobRequest, Settings, RunCallbacks], WorkflowExecutionResult],
    ) -> None:
        self._config_path = config_path
        self._auto_runner = auto_runner
        self._manual_runner = manual_runner
        self._lock = threading.Lock()
        self._current_job: _Job | None = None
        self._latest_job: _Job | None = None

    def defaults(self) -> dict[str, Any]:
        settings = load_config(self._config_path)
        latest_worker_image = _load_latest_known_worker_image()
        return {
            "target_width": settings.video.target_width,
            "target_height": settings.video.target_height,
            "gpu_models": settings.vast.gpu_models,
            "worker_image_latest_known": latest_worker_image,
            "worker_image_source": "history" if latest_worker_image else "none",
            "worker_image_required": latest_worker_image == "",
            "ssh_bootstrap_mode": "tofu",
        }

    def list_instances(self) -> list[dict[str, Any]]:
        settings = load_config(self._config_path)
        client = _build_write_client(settings=settings)
        try:
            rows = client.list_instances(limit=25)
        finally:
            client.close()
        return rows

    def start_job(self, request: StartJobRequest) -> dict[str, Any]:
        defaults = self.defaults()
        self._validate_paths(request=request)
        job = _Job(request=request, defaults=defaults)
        with self._lock:
            if self._current_job is not None and self._current_job.is_active():
                raise HTTPException(status_code=409, detail="Another processing task is already active.")
            self._current_job = job
            self._latest_job = job
        job.append_event(
            "status",
            {
                "state": "pending",
                "mode": request.mode,
                "message": "Task created. Waiting for worker thread startup.",
            },
        )
        job.append_event(
            "configuration",
            {
                "mode": request.mode,
                "input_path": request.input_path,
                "output_path": str(_resolve_output_path(request=request)),
                "report_path": str(_resolve_report_path(request=request, output_path=_resolve_output_path(request=request))),
                "target_width": request.target_width,
                "target_height": request.target_height,
                "gpu_model": request.gpu_model,
                "worker_image_requested": (request.worker_image or "").strip() or None,
                "offer_id": request.offer_id,
                "instance_id": request.instance_id,
            },
        )
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        job.thread = thread
        thread.start()
        return self._job_snapshot(job)

    def get_job(self, job_id: str) -> _Job:
        candidates = [self._current_job, self._latest_job]
        for item in candidates:
            if item is not None and item.job_id == job_id:
                return item
        raise HTTPException(status_code=404, detail="Job not found.")

    def job_snapshot(self, job_id: str) -> dict[str, Any]:
        return self._job_snapshot(self.get_job(job_id))

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        with job.condition:
            if not job.is_active():
                return self._job_snapshot(job)
            job.cancel_requested = True
            if job.pending_prompt is not None:
                job.pending_prompt["response"] = {"approved": False}
                job.pending_prompt["event"].set()
            job.condition.notify_all()
        job.append_event("status", {"state": "cancel_requested", "message": "Cancellation requested by operator."})
        return self._job_snapshot(job)

    def answer_prompt(self, job_id: str, prompt_id: str, reply: PromptReplyRequest) -> dict[str, Any]:
        job = self.get_job(job_id)
        with job.condition:
            pending = job.pending_prompt
            if prompt_id in job.answered_prompt_ids:
                raise HTTPException(status_code=409, detail="Prompt already resolved.")
            if pending is None:
                raise HTTPException(status_code=404, detail="Prompt not found.")
            if pending["id"] != prompt_id:
                raise HTTPException(status_code=409, detail="Another prompt is currently pending.")
            pending["response"] = {"approved": reply.approved, "fingerprint": (reply.fingerprint or "").strip()}
            pending["event"].set()
            job.answered_prompt_ids.add(prompt_id)
            job.pending_prompt = None
            job.condition.notify_all()
        job.append_event("prompt-answer", {"prompt_id": prompt_id, "approved": reply.approved})
        return self._job_snapshot(job)

    def stream_events(self, job_id: str, last_event_id: int):
        job = self.get_job(job_id)

        def _generator():
            cursor = max(0, last_event_id)
            while True:
                with job.condition:
                    pending = [e for e in job.events if int(e["id"]) > cursor]
                    if not pending:
                        if not job.is_active():
                            break
                        job.condition.wait(timeout=15.0)
                        continue
                for event in pending:
                    cursor = int(event["id"])
                    payload = json.dumps(event, separators=(",", ":"))
                    yield f"id: {cursor}\ndata: {payload}\n\n"
            yield "event: end\ndata: {}\n\n"

        return _generator()

    def _run_job(self, job: _Job) -> None:
        job.status = "running"
        job.append_event("status", {"state": "running", "message": "Workflow started."})
        started = time.perf_counter()
        settings = load_config(self._config_path)

        callbacks = RunCallbacks(
            report_provisioning_status=lambda payload: self._on_provisioning(job, payload),
            report_worker_progress=lambda payload: self._on_worker_progress(job, payload),
            confirm_paid=lambda message: self._await_boolean_prompt(
                job=job,
                kind="paid_confirmation",
                message=message,
            ),
            confirm_ssh_trust=lambda message: self._await_boolean_prompt(
                job=job,
                kind="ssh_trust",
                message=message,
            ),
            request_verified_fingerprint=lambda message: self._await_fingerprint_prompt(job=job, message=message),
            raise_if_cancelled=lambda: self._raise_if_cancelled(job),
        )

        try:
            if job.request.mode == "automatic":
                result = self._auto_runner(job.request, settings, callbacks)
            else:
                result = self._manual_runner(job.request, settings, callbacks)
            elapsed = time.perf_counter() - started
            if elapsed > result.duration_seconds:
                result = WorkflowExecutionResult(
                    mode=result.mode,
                    session_id=result.session_id,
                    state=result.state,
                    output_path=result.output_path,
                    report_path=result.report_path,
                    duration_seconds=elapsed,
                    estimated_total_usd=result.estimated_total_usd,
                    instance_id=result.instance_id,
                    instance_state=result.instance_state,
                    cleanup_verified=result.cleanup_verified,
                    cleanup_warning=result.cleanup_warning,
                )
            job.result = result
            job.status = "completed"
            if job.request.worker_image:
                _store_latest_worker_image(job.request.worker_image)
            job.append_event(
                "result",
                {
                    "state": result.state,
                    "output_path": result.output_path,
                    "report_path": result.report_path,
                    "duration_seconds": round(result.duration_seconds, 3),
                    "estimated_total_usd": result.estimated_total_usd,
                    "instance_id": result.instance_id,
                    "instance_state": result.instance_state,
                    "cleanup_verified": result.cleanup_verified,
                    "cleanup_warning": result.cleanup_warning,
                },
            )
        except WebWorkflowCancelled:
            job.status = "cancelled"
            job.error = "Cancelled by operator."
            job.append_event("error", {"message": job.error})
            if job.request.mode == "manual":
                job.append_event(
                    "status",
                    {"state": "manual-instance-preserved", "message": "Manual mode never destroys the existing instance automatically."},
                )
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            job.append_event("error", {"message": str(exc)})
            if job.request.mode == "manual":
                job.append_event(
                    "status",
                    {"state": "manual-instance-preserved", "message": "Manual mode never destroys the existing instance automatically."},
                )
        finally:
            with self._lock:
                if self._current_job is not None and self._current_job.job_id == job.job_id:
                    self._current_job = job
            with job.condition:
                job.condition.notify_all()

    def _on_provisioning(self, job: _Job, payload: dict[str, str]) -> None:
        self._raise_if_cancelled(job)
        job.append_event("provisioning", payload)

    def _on_worker_progress(self, job: _Job, payload: dict[str, object]) -> None:
        self._raise_if_cancelled(job)
        job.append_event("worker", payload)

    def _await_boolean_prompt(self, *, job: _Job, kind: str, message: str) -> bool:
        response = self._await_prompt(job=job, kind=kind, message=message)
        approved = response.get("approved")
        if approved is not True:
            if job.cancel_requested:
                raise WebWorkflowCancelled()
            return False
        return True

    def _await_fingerprint_prompt(self, *, job: _Job, message: str) -> str | None:
        response = self._await_prompt(job=job, kind="verified_fingerprint", message=message)
        fingerprint = (response.get("fingerprint") or "").strip()
        if not fingerprint:
            return None
        return fingerprint

    def _await_prompt(self, *, job: _Job, kind: str, message: str) -> dict[str, Any]:
        self._raise_if_cancelled(job)
        prompt_id = f"prompt-{uuid4().hex[:8]}"
        wait_event = threading.Event()
        pending = {
            "id": prompt_id,
            "kind": kind,
            "message": message,
            "response": None,
            "event": wait_event,
        }
        with job.condition:
            job.pending_prompt = pending
            job.condition.notify_all()
        job.append_event("prompt", {"prompt_id": prompt_id, "kind": kind, "message": message})
        while True:
            if wait_event.wait(timeout=0.5):
                break
            self._raise_if_cancelled(job)
        response = pending["response"]
        if not isinstance(response, dict):
            raise WebWorkflowCancelled()
        return response

    @staticmethod
    def _raise_if_cancelled(job: _Job) -> None:
        if job.cancel_requested:
            raise WebWorkflowCancelled()

    @staticmethod
    def _job_snapshot(job: _Job) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "mode": job.request.mode,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "error": job.error,
            "pending_prompt": None,
        }
        if job.pending_prompt is not None:
            payload["pending_prompt"] = {
                "prompt_id": job.pending_prompt["id"],
                "kind": job.pending_prompt["kind"],
                "message": job.pending_prompt["message"],
            }
        if job.result is not None:
            payload["result"] = {
                "mode": job.result.mode,
                "session_id": job.result.session_id,
                "state": job.result.state,
                "output_path": job.result.output_path,
                "report_path": job.result.report_path,
                "duration_seconds": round(job.result.duration_seconds, 3),
                "estimated_total_usd": job.result.estimated_total_usd,
                "instance_id": job.result.instance_id,
                "instance_state": job.result.instance_state,
                "cleanup_verified": job.result.cleanup_verified,
                "cleanup_warning": job.result.cleanup_warning,
            }
        return payload

    @staticmethod
    def _validate_paths(*, request: StartJobRequest) -> None:
        input_path = Path(request.input_path)
        if not input_path.exists() or not input_path.is_file():
            raise HTTPException(status_code=400, detail="Input path must be an existing file.")
        output_path = _resolve_output_path(request=request)
        report_path = _resolve_report_path(request=request, output_path=output_path)
        if output_path.exists():
            raise HTTPException(status_code=400, detail=f"Refusing to overwrite existing output: {output_path}")
        if report_path.exists():
            raise HTTPException(status_code=400, detail=f"Refusing to overwrite existing report: {report_path}")
        if output_path.resolve() == report_path.resolve():
            raise HTTPException(status_code=400, detail="Output path and report path must be different.")
        if not output_path.parent.exists():
            raise HTTPException(status_code=400, detail=f"Output directory does not exist: {output_path.parent}")
        if not report_path.parent.exists():
            raise HTTPException(status_code=400, detail=f"Report directory does not exist: {report_path.parent}")
        if request.mode == "automatic":
            image = (request.worker_image or _load_latest_known_worker_image()).strip()
            if not image:
                raise HTTPException(
                    status_code=400,
                    detail="Automatic mode requires an explicit worker image tag (no placeholder/default available).",
                )
            if image.lower() in _WORKER_IMAGE_PLACEHOLDERS:
                raise HTTPException(status_code=400, detail=f"Refusing placeholder worker image tag: {image}")


def _default_auto_runner(request: StartJobRequest, settings: Settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
    callbacks.raise_if_cancelled()
    effective_settings = settings.model_copy(deep=True)
    if request.gpu_model:
        effective_settings.vast.gpu_models = [request.gpu_model]

    output_path = _resolve_output_path(request=request)
    report_path = _resolve_report_path(request=request, output_path=output_path)
    worker_image = (request.worker_image or _load_latest_known_worker_image()).strip()
    if not worker_image:
        raise OrchestrationError("Worker image tag is required.")
    if worker_image.lower() in _WORKER_IMAGE_PLACEHOLDERS:
        raise OrchestrationError(f"Refusing placeholder worker image tag: {worker_image}")
    callbacks.report_provisioning_status(
        {
            "phase": "worker-image",
            "instance_id": "n/a",
            "gpu": request.gpu_model or "n/a",
            "host": "n/a",
            "status": "selected",
            "status_message": f"Worker image: {worker_image}",
        }
    )

    read_client = _build_read_client(effective_settings)
    write_client = _build_write_client(effective_settings)
    transport = OpenSshTransport()
    store = SessionStore()

    try:
        with SessionExecutionLock():
            summary = run_upscale_workflow(
                settings=effective_settings,
                options=OrchestrationOptions(
                    input_path=Path(request.input_path),
                    output_path=output_path,
                    report_path=report_path,
                    worker_image=worker_image,
                    profile=effective_settings.video.default_profile,
                    target_width=request.target_width or effective_settings.video.target_width,
                    target_height=request.target_height or effective_settings.video.target_height,
                    keep_instance=False,
                    processing_fps=request.processing_fps,
                    sort_by=request.sort_by,
                    offer_id=request.offer_id,
                    ssh_bootstrap_mode=request.ssh_bootstrap_mode,
                ),
                session_store=store,
                read_client=read_client,
                write_client=write_client,
                transport=transport,
                confirm_paid=lambda message: callbacks.confirm_paid(message),
                confirm_ssh_trust=lambda message: callbacks.confirm_ssh_trust(message),
                request_verified_fingerprint=lambda message: callbacks.request_verified_fingerprint(message),
                report_provisioning_status=lambda payload: callbacks.report_provisioning_status(payload),
                report_worker_progress=lambda payload: callbacks.report_worker_progress(payload),
            )
    finally:
        read_client.close()
        write_client.close()

    return WorkflowExecutionResult(
        mode="automatic",
        session_id=summary.session_id,
        state=summary.state,
        output_path=str(summary.worker_result.local_output_path) if summary.worker_result is not None else None,
        report_path=str(summary.worker_result.local_report_path) if summary.worker_result is not None else str(report_path),
        duration_seconds=0.0,
        estimated_total_usd=summary.estimate.total_estimated_usd,
        instance_id=summary.instance_id,
        instance_state=summary.state,
        cleanup_verified=summary.cleanup_verified,
        cleanup_warning=summary.cleanup_warning,
    )


def _default_manual_runner(request: StartJobRequest, settings: Settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
    callbacks.raise_if_cancelled()
    if request.instance_id is None:
        raise OrchestrationError("Manual mode requires an existing Vast instance id.")

    output_path = _resolve_output_path(request=request)
    report_path = _resolve_report_path(request=request, output_path=output_path)

    transport = OpenSshTransport()
    write_client = _build_write_client(settings)
    started = time.perf_counter()
    callbacks.report_provisioning_status(
        {
            "phase": "worker-image",
            "instance_id": str(request.instance_id or "n/a"),
            "gpu": "unknown",
            "host": "unknown",
            "status": "manual-instance",
            "status_message": "Manual mode uses the image already present on the selected instance.",
        }
    )
    try:
        instance = write_client.get_instance(request.instance_id)
        status = (instance.actual_status or "").lower()
        if status != "running":
            raise OrchestrationError(
                f"Manual mode requires a running instance. Current status: {instance.actual_status or 'unknown'}."
            )
        tracker = ProvisioningAttemptTracker(session_id=f"manual-{uuid4().hex[:8]}")
        _wait_for_ssh_tcp_availability(
            instance=instance,
            timeout_seconds=75.0,
            retry_delay_seconds=5.0,
            report_provisioning_status=lambda payload: callbacks.report_provisioning_status(payload),
            tracker=tracker,
        )
        callbacks.raise_if_cancelled()
        connection = _build_trusted_connection(
            instance=instance,
            private_key_path=_resolve_private_key(
                settings=settings,
                options=OrchestrationOptions(
                    input_path=Path(request.input_path),
                    output_path=output_path,
                    report_path=report_path,
                    worker_image=request.worker_image or _load_latest_known_worker_image(),
                    profile=settings.video.default_profile,
                    target_width=request.target_width or settings.video.target_width,
                    target_height=request.target_height or settings.video.target_height,
                    ssh_bootstrap_mode=request.ssh_bootstrap_mode,
                ),
            ),
            known_hosts_path=Path.home() / ".ssh" / "known_hosts",
            confirm_ssh_trust=lambda message: callbacks.confirm_ssh_trust(message),
            bootstrap_mode=request.ssh_bootstrap_mode,
            request_verified_fingerprint=lambda message: callbacks.request_verified_fingerprint(message),
        )
        transport.check_ssh_ready(connection, timeout_seconds=30.0)
        callbacks.report_provisioning_status(
            {
                "phase": "manual-warning",
                "instance_id": str(instance.instance_id),
                "gpu": instance.gpu_name or "unknown",
                "host": instance.ssh_host or "unknown",
                "status": "billing-continues",
                "status_message": "Manual mode does not destroy the instance automatically. Billing continues.",
            }
        )
        worker_result = execute_remote_worker(
            transport=transport,
            connection=connection,
            request=RemoteWorkerRequest(
                session_id=f"manual-{uuid4().hex[:12]}",
                local_input_path=Path(request.input_path),
                local_output_path=output_path,
                local_report_path=report_path,
                target_width=request.target_width or settings.video.target_width,
                target_height=request.target_height or settings.video.target_height,
                profile=settings.video.default_profile,
            ),
            progress_callback=lambda payload: callbacks.report_worker_progress(payload),
        )
        elapsed = time.perf_counter() - started
        return WorkflowExecutionResult(
            mode="manual",
            session_id=None,
            state="COMPLETED",
            output_path=str(worker_result.local_output_path),
            report_path=str(worker_result.local_report_path),
            duration_seconds=elapsed,
            estimated_total_usd=None,
            instance_id=instance.instance_id,
            instance_state=instance.actual_status,
            cleanup_verified=False,
            cleanup_warning="Manual mode leaves instance running. Destroy manually when done.",
        )
    finally:
        write_client.close()


def _build_read_client(settings: Settings) -> VastClient:
    api_key = settings.secrets.vast_api_key
    if not api_key:
        raise OrchestrationError("VAST_API_KEY missing.")
    return VastClient(api_key=api_key, config=VastClientConfig())


def _build_write_client(settings: Settings) -> VastWriteClient:
    api_key = settings.secrets.vast_api_key
    if not api_key:
        raise OrchestrationError("VAST_API_KEY missing.")
    return VastWriteClient(api_key=api_key)


def _resolve_output_path(*, request: StartJobRequest) -> Path:
    if request.output_path:
        return Path(request.output_path)
    source = Path(request.input_path)
    suffix = source.suffix or ".mkv"
    return source.with_name(f"{source.stem}-upscaled{suffix}")


def _resolve_report_path(*, request: StartJobRequest, output_path: Path) -> Path:
    if request.report_path:
        return Path(request.report_path)
    return output_path.with_suffix(output_path.suffix + ".report.json")


def _load_latest_known_worker_image() -> str:
    try:
        payload = json.loads(_WORKER_IMAGE_HISTORY_PATH.read_text(encoding="utf-8"))
    except OSError:
        return _DEFAULT_WORKER_IMAGE
    except json.JSONDecodeError:
        return _DEFAULT_WORKER_IMAGE
    if not isinstance(payload, dict):
        return _DEFAULT_WORKER_IMAGE
    value = payload.get("latest")
    if isinstance(value, str):
        normalized = value.strip()
        if normalized and normalized.lower() not in _WORKER_IMAGE_PLACEHOLDERS:
            return normalized
    return _DEFAULT_WORKER_IMAGE


def _store_latest_worker_image(value: str) -> None:
    image = value.strip()
    if not image:
        return
    if image.lower() in _WORKER_IMAGE_PLACEHOLDERS:
        return
    _WORKER_IMAGE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WORKER_IMAGE_HISTORY_PATH.write_text(json.dumps({"latest": image}, indent=2), encoding="utf-8")


def create_app(
    *,
    config_path: Path | None = None,
    auto_runner: Callable[[StartJobRequest, Settings, RunCallbacks], WorkflowExecutionResult] | None = None,
    manual_runner: Callable[[StartJobRequest, Settings, RunCallbacks], WorkflowExecutionResult] | None = None,
) -> FastAPI:
    resolved_config = (config_path or Path("config.yaml")).resolve()
    manager = WebTaskManager(
        config_path=resolved_config,
        auto_runner=auto_runner or _default_auto_runner,
        manual_runner=manual_runner or _default_manual_runner,
    )
    app = FastAPI(title="Vast Upscaler Local Web", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return manager.defaults()

    @app.post("/api/video/analyze")
    def api_video_analyze(request: SourceAnalysisRequest) -> dict[str, Any]:
        try:
            if not request.include_visual_diagnostics:
                analysis = analyze_source_video(Path(request.input_path))
                return analysis.model_dump(mode="json")
            return analyze_source_video_report(
                video_path=Path(request.input_path),
                include_visual_diagnostics=True,
                diagnostics_options=request.diagnostic_options,
            )
        except SourceAnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/video/preprocessing-plan")
    def api_preprocessing_plan(request: PreprocessingPlanRequest) -> dict[str, Any]:
        source_count = sum(
            value is not None and value != ""
            for value in (request.input_path, request.analysis_payload, request.analysis_report_path)
        )
        if source_count != 1:
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one source: input_path, analysis_payload, or analysis_report_path.",
            )
        try:
            if request.analysis_payload is not None:
                analysis_payload = request.analysis_payload
            elif request.analysis_report_path is not None:
                analysis_payload = load_analysis_payload_from_file(Path(request.analysis_report_path))
            else:
                analysis_payload = analyze_source_video_report(
                    video_path=Path(request.input_path or ""),
                    include_visual_diagnostics=request.include_visual_diagnostics,
                    diagnostics_options=request.diagnostic_options,
                )
            plan = build_preprocessing_plan(
                analysis_payload=analysis_payload,
                user_choices=request.user_choices or PlannerUserChoices(),
            )
        except (SourceAnalysisError, PreprocessingPlannerError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return plan.model_dump(mode="json")

    @app.get("/api/instances")
    def api_instances() -> dict[str, Any]:
        return {"instances": manager.list_instances()}

    @app.post("/api/jobs/start")
    def api_start(request: StartJobRequest) -> dict[str, Any]:
        return manager.start_job(request)

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> dict[str, Any]:
        return manager.job_snapshot(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel(job_id: str) -> dict[str, Any]:
        return manager.cancel(job_id)

    @app.post("/api/jobs/{job_id}/prompts/{prompt_id}")
    def api_prompt_answer(job_id: str, prompt_id: str, reply: PromptReplyRequest) -> dict[str, Any]:
        return manager.answer_prompt(job_id=job_id, prompt_id=prompt_id, reply=reply)

    @app.get("/api/jobs/{job_id}/events")
    def api_events(job_id: str, last_event_id: int = 0) -> StreamingResponse:
        stream = manager.stream_events(job_id=job_id, last_event_id=last_event_id)
        return StreamingResponse(stream, media_type="text/event-stream")

    return app


_INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Vast Upscaler Local Web</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    fieldset { margin-bottom: 14px; }
    label { display: block; margin-top: 8px; }
    input, select { width: 100%; padding: 6px; margin-top: 2px; }
    button { padding: 8px 12px; margin-right: 8px; }
    #events { background: #111; color: #ddd; padding: 10px; white-space: pre-wrap; height: 300px; overflow: auto; }
    .warn { color: #9a6700; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Vast Upscaler (local)</h1>
  <p class="warn">Runs only on 127.0.0.1. API keys are never sent to the browser.</p>
  <form id="job-form">
    <fieldset>
      <legend>Configuration</legend>
      <label>Mode
        <select id="mode">
          <option value="automatic">Automatic (create + cleanup)</option>
          <option value="manual">Manual (existing instance, no auto-destroy)</option>
        </select>
      </label>
      <label>Input video path <input id="input_path" required /></label>
      <label>Output path (optional) <input id="output_path" /></label>
      <label>Report path (optional) <input id="report_path" /></label>
      <label>Target width <input id="target_width" type="number" min="1" /></label>
      <label>Target height <input id="target_height" type="number" min="1" /></label>
      <label>GPU model
        <select id="gpu_model"></select>
      </label>
      <label>Worker image tag <input id="worker_image" /></label>
      <label>Sort offers by
        <select id="sort_by">
          <option value="price">price</option>
          <option value="reliability">reliability</option>
          <option value="performance">performance</option>
        </select>
      </label>
      <label>Offer ID (automatic mode optional) <input id="offer_id" type="number" /></label>
      <label>Instance ID (manual mode required) <input id="instance_id" type="number" /></label>
      <label>SSH bootstrap mode
        <select id="ssh_bootstrap_mode">
          <option value="tofu">tofu</option>
          <option value="strict">strict</option>
        </select>
      </label>
    </fieldset>
    <button type="submit">Start</button>
    <button type="button" id="cancel-btn">Cancel</button>
  </form>

  <h2>Status</h2>
  <div id="status">idle</div>
  <h2>Events</h2>
  <div id="events"></div>

  <script>
    let currentJobId = null;
    let lastEventId = 0;
    let es = null;
    let promptDialogOpen = false;
    const handledPromptIds = new Set();

    function appendEvent(msg) {
      const box = document.getElementById("events");
      box.textContent += msg + "\\n";
      box.scrollTop = box.scrollHeight;
    }

    async function loadDefaults() {
      const res = await fetch("/api/config");
      const cfg = await res.json();
      document.getElementById("target_width").value = cfg.target_width;
      document.getElementById("target_height").value = cfg.target_height;
      document.getElementById("worker_image").value = cfg.worker_image_latest_known;
      if (cfg.worker_image_required) {
        appendEvent("No reliable worker image default is known. Enter an explicit real tag before automatic mode.");
      }
      const gpu = document.getElementById("gpu_model");
      gpu.innerHTML = "";
      for (const model of cfg.gpu_models) {
        const opt = document.createElement("option");
        opt.value = model;
        opt.textContent = model;
        gpu.appendChild(opt);
      }
      appendEvent("Default worker image (latest known): " + cfg.worker_image_latest_known);
    }

    async function answerPrompt(jobId, prompt) {
      if (handledPromptIds.has(prompt.prompt_id)) return;
      handledPromptIds.add(prompt.prompt_id);
      promptDialogOpen = true;
      if (prompt.kind === "verified_fingerprint") {
        const fp = window.prompt(prompt.message + "\\n\\nEnter independently verified SHA256 fingerprint:");
        await fetch(`/api/jobs/${jobId}/prompts/${prompt.prompt_id}`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ fingerprint: fp || "" }),
        });
        promptDialogOpen = false;
        return;
      }
      const approved = window.confirm(prompt.message);
      await fetch(`/api/jobs/${jobId}/prompts/${prompt.prompt_id}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ approved }),
      });
      promptDialogOpen = false;
    }

    async function maybeHandlePrompt(jobId, prompt) {
      if (promptDialogOpen) return;
      const snapshotResp = await fetch(`/api/jobs/${jobId}`);
      if (!snapshotResp.ok) return;
      const snapshot = await snapshotResp.json();
      const pending = snapshot.pending_prompt;
      if (!pending || pending.prompt_id !== prompt.prompt_id) return;
      await answerPrompt(jobId, prompt);
    }

    function watchEvents(jobId) {
      if (es) es.close();
      es = new EventSource(`/api/jobs/${jobId}/events?last_event_id=${lastEventId}`);
      es.onmessage = async (evt) => {
        const data = JSON.parse(evt.data);
        lastEventId = data.id;
        appendEvent(`[${data.type}] ${JSON.stringify(data.payload)}`);
        if (data.type === "prompt") {
          await maybeHandlePrompt(jobId, data.payload);
        }
        if (data.type === "result" || data.type === "error") {
          document.getElementById("status").textContent = data.type === "result" ? "completed" : "failed";
        }
      };
    }

    document.getElementById("job-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = {
        mode: document.getElementById("mode").value,
        input_path: document.getElementById("input_path").value,
        output_path: document.getElementById("output_path").value || null,
        report_path: document.getElementById("report_path").value || null,
        target_width: Number(document.getElementById("target_width").value),
        target_height: Number(document.getElementById("target_height").value),
        gpu_model: document.getElementById("gpu_model").value || null,
        worker_image: document.getElementById("worker_image").value || null,
        sort_by: document.getElementById("sort_by").value,
        offer_id: document.getElementById("offer_id").value ? Number(document.getElementById("offer_id").value) : null,
        instance_id: document.getElementById("instance_id").value ? Number(document.getElementById("instance_id").value) : null,
        ssh_bootstrap_mode: document.getElementById("ssh_bootstrap_mode").value,
      };
      const response = await fetch("/api/jobs/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        appendEvent("Start failed: " + (data.detail || JSON.stringify(data)));
        return;
      }
      currentJobId = data.job_id;
      lastEventId = 0;
      handledPromptIds.clear();
      promptDialogOpen = false;
      document.getElementById("status").textContent = "running";
      appendEvent("Job started: " + currentJobId);
      watchEvents(currentJobId);
    });

    document.getElementById("cancel-btn").addEventListener("click", async () => {
      if (!currentJobId) return;
      await fetch(`/api/jobs/${currentJobId}/cancel`, {method: "POST"});
      appendEvent("Cancel requested.");
    });

    loadDefaults();
  </script>
</body>
</html>
"""
