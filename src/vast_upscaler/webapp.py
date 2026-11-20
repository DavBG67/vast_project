from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from vast_upscaler.config import Settings, load_config
from vast_upscaler.local_preprocessing import (
    LocalEncodingOptions,
    LocalExecutionOptions,
    LocalPreprocessingError,
    execute_preprocessing_plan,
)
from vast_upscaler.orchestrator import (
    OrchestrationError,
    OrchestrationOptions,
    SessionExecutionLock,
    _build_trusted_connection,
    _resolve_private_key,
    _wait_for_ssh_tcp_availability,
    run_upscale_workflow,
)
from vast_upscaler.preprocessing_contract import (
    RemotePreprocessingContractError,
    RemotePreprocessingContractValidationError,
    build_remote_preprocessing_contract_from_plan_payload,
    dump_remote_preprocessing_contract_payload,
    load_preprocessing_plan_payload,
)
from vast_upscaler.preprocessing_planner import (
    PlannerUserChoices,
    PreprocessingPlan,
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
_VIDEO_SOURCE_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
}
_VIDEO_SOURCE_EXCLUDED_SUFFIXES = ("-upscaled", "-cropped")


class WebWorkflowCancelled(RuntimeError):
    pass


class StartJobRequest(BaseModel):
    mode: RunMode
    input_path: str | None = None
    source_name: str | None = None
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
    preprocessing_plan_path: str | None = None
    preprocessing_plan_payload: dict[str, Any] | None = None
    ssh_bootstrap_mode: SshBootstrapMode = "tofu"


class PromptReplyRequest(BaseModel):
    approved: bool | None = None
    fingerprint: str | None = None


class SourceAnalysisRequest(BaseModel):
    input_path: str | None = None
    source_name: str | None = None
    include_visual_diagnostics: bool = False
    diagnostic_options: VisualDiagnosticsOptions | None = None


class PreprocessingPlanRequest(BaseModel):
    input_path: str | None = None
    analysis_payload: dict[str, Any] | None = None
    analysis_report_path: str | None = None
    include_visual_diagnostics: bool = False
    diagnostic_options: VisualDiagnosticsOptions | None = None
    user_choices: PlannerUserChoices | None = None


class PreprocessingExecutionPreviewRequest(BaseModel):
    plan_payload: dict[str, Any]
    output_path: str
    video_codec: str = "libx264"
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "medium"
    pixel_format: str = "yuv420p"


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
                "input_path": str(_resolve_input_video_path(request=request)),
                "source_name": request.source_name,
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
        source_name = (request.source_name or "").strip()
        input_path_raw = (request.input_path or "").strip()
        if source_name and input_path_raw:
            raise HTTPException(status_code=400, detail="Provide either source_name or input_path, not both.")
        if not source_name and not input_path_raw:
            raise HTTPException(status_code=400, detail="Either source_name or input_path is required.")

        input_path = _resolve_input_video_path(request=request)
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
        if request.preprocessing_plan_path is not None:
            plan_path = Path(request.preprocessing_plan_path)
            if not plan_path.exists() or not plan_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Preprocessing plan path must be an existing file: {plan_path}",
                )
        if request.preprocessing_plan_payload is not None and not isinstance(request.preprocessing_plan_payload, dict):
            raise HTTPException(status_code=400, detail="preprocessing_plan_payload must be a JSON object.")
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
    input_path = _resolve_input_video_path(request=request)
    preprocessing_contract = _resolve_remote_preprocessing_contract(request=request)
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
                    input_path=input_path,
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
                    preprocessing_contract=preprocessing_contract,
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
    input_path = _resolve_input_video_path(request=request)
    preprocessing_contract = _resolve_remote_preprocessing_contract(request=request)

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
                    input_path=input_path,
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
                local_input_path=input_path,
                local_output_path=output_path,
                local_report_path=report_path,
                target_width=request.target_width or settings.video.target_width,
                target_height=request.target_height or settings.video.target_height,
                profile=settings.video.default_profile,
                preprocessing_contract=preprocessing_contract,
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
    source = _resolve_input_video_path(request=request)
    suffix = source.suffix or ".mkv"
    return source.with_name(f"{source.stem}-upscaled{suffix}")


def _resolve_report_path(*, request: StartJobRequest, output_path: Path) -> Path:
    if request.report_path:
        return Path(request.report_path)
    return output_path.with_suffix(output_path.suffix + ".report.json")


def _resolve_remote_preprocessing_contract(*, request: StartJobRequest) -> dict[str, Any] | None:
    try:
        if request.preprocessing_plan_path is not None:
            plan_path = Path(request.preprocessing_plan_path)
            payload = load_preprocessing_plan_payload(plan_path)
            contract = build_remote_preprocessing_contract_from_plan_payload(payload)
            return dump_remote_preprocessing_contract_payload(contract)
        if request.preprocessing_plan_payload is not None:
            contract = build_remote_preprocessing_contract_from_plan_payload(request.preprocessing_plan_payload)
            return dump_remote_preprocessing_contract_payload(contract)
    except (RemotePreprocessingContractError, RemotePreprocessingContractValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid preprocessing plan for remote transport: {exc}") from exc
    return None


def _project_root_path() -> Path:
    return Path(__file__).resolve().parents[2]


def _video_sources_dir() -> Path:
    return _project_root_path() / "video"


def _normalize_source_name(source_name: str) -> str:
    value = source_name.strip()
    if not value:
        raise HTTPException(status_code=400, detail="Video source name must be non-empty.")
    if "\\" in value or "/" in value:
        raise HTTPException(status_code=400, detail="Video source name must not include path separators.")
    if PurePath(value).name != value:
        raise HTTPException(status_code=400, detail="Video source name is invalid.")
    return value


def _is_excluded_generated_source(name: str) -> bool:
    lower_name = name.lower()
    if lower_name.endswith(".report.json"):
        return True
    stem = Path(lower_name).stem
    return stem.endswith(_VIDEO_SOURCE_EXCLUDED_SUFFIXES)


def _resolve_video_source_path(*, source_name: str) -> Path:
    normalized_name = _normalize_source_name(source_name)
    suffix = Path(normalized_name).suffix.lower()
    if suffix not in _VIDEO_SOURCE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video extension for source '{normalized_name}'.")
    if _is_excluded_generated_source(normalized_name):
        raise HTTPException(status_code=400, detail=f"Source '{normalized_name}' is excluded from selectable inputs.")

    sources_dir = _video_sources_dir()
    candidate = sources_dir / normalized_name
    try:
        resolved_sources = sources_dir.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Video sources directory is unavailable: {sources_dir}") from exc
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        raise HTTPException(status_code=404, detail=f"Video source not found: {normalized_name}") from None
    if not resolved_candidate.is_file():
        raise HTTPException(status_code=400, detail=f"Video source is not a file: {normalized_name}")
    try:
        resolved_candidate.relative_to(resolved_sources)
    except ValueError:
        raise HTTPException(status_code=400, detail="Video source resolves outside the allowed directory.") from None
    return resolved_candidate


def _resolve_input_video_path(*, request: StartJobRequest) -> Path:
    source_name = (request.source_name or "").strip()
    input_path = (request.input_path or "").strip()
    if source_name:
        return _resolve_video_source_path(source_name=source_name)
    if not input_path:
        raise HTTPException(status_code=400, detail="Either source_name or input_path must be provided.")
    return Path(input_path)


def _list_video_sources_payload() -> dict[str, Any]:
    sources_dir = _video_sources_dir()
    sources: list[dict[str, Any]] = []
    if not sources_dir.exists():
        return {"directory": str(sources_dir), "sources": sources}
    if not sources_dir.is_dir():
        raise HTTPException(status_code=500, detail=f"Video sources path is not a directory: {sources_dir}")

    for entry in sorted(sources_dir.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _VIDEO_SOURCE_EXTENSIONS:
            continue
        if _is_excluded_generated_source(entry.name):
            continue
        try:
            resolved = _resolve_video_source_path(source_name=entry.name)
        except HTTPException:
            continue
        stat = resolved.stat()
        sources.append(
            {
                "name": entry.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            }
        )
    return {"directory": str(sources_dir), "sources": sources}


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

    @app.get("/api/video/sources")
    def api_video_sources() -> dict[str, Any]:
        return _list_video_sources_payload()

    @app.post("/api/video/analyze")
    def api_video_analyze(request: SourceAnalysisRequest) -> dict[str, Any]:
        source_count = sum(value is not None and value != "" for value in (request.input_path, request.source_name))
        if source_count != 1:
            raise HTTPException(status_code=400, detail="Provide exactly one source: input_path or source_name.")
        try:
            if request.source_name:
                video_path = _resolve_video_source_path(source_name=request.source_name)
            else:
                video_path = Path(request.input_path or "")
            if not request.include_visual_diagnostics:
                analysis = analyze_source_video(video_path)
                return analysis.model_dump(mode="json")
            return analyze_source_video_report(
                video_path=video_path,
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

    @app.post("/api/video/preprocessing-execution/preview")
    def api_preprocessing_execution_preview(request: PreprocessingExecutionPreviewRequest) -> dict[str, Any]:
        try:
            plan = PreprocessingPlan.model_validate(request.plan_payload)
            report = execute_preprocessing_plan(
                plan=plan,
                output_path=Path(request.output_path),
                encoding=LocalEncodingOptions(
                    video_codec=request.video_codec,
                    crf=request.crf,
                    preset=request.preset,
                    pixel_format=request.pixel_format,
                ),
                options=LocalExecutionOptions(dry_run=True),
            )
        except (LocalPreprocessingError, PreprocessingPlannerError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.model_dump(mode="json")

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
    details { margin-top: 10px; }
    label { display: block; margin-top: 8px; }
    input, select { width: 100%; padding: 6px; margin-top: 2px; }
    button { padding: 8px 12px; margin-right: 8px; margin-top: 8px; }
    #events { background: #111; color: #ddd; padding: 10px; white-space: pre-wrap; height: 300px; overflow: auto; }
    .warn { color: #9a6700; font-weight: bold; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip { background: #f3f3f3; border-radius: 12px; padding: 2px 8px; font-size: 12px; }
    .muted { color: #555; font-size: 13px; }
    .status-line { margin-top: 6px; font-weight: bold; }
    .hidden { display: none; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }
    code { white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body>
  <h1>Vast Upscaler (local)</h1>
  <p class="warn">Runs only on 127.0.0.1. API keys are never sent to the browser.</p>
  <form id="job-form">
    <fieldset>
      <legend>1) Choose video</legend>
      <label>Source video (project video/ folder)
        <select id="source_name">
          <option value="">-- Select a video --</option>
        </select>
      </label>
      <div class="chips">
        <button type="button" id="refresh-sources-btn">Refresh</button>
      </div>
      <div id="video-sources-dir" class="muted"></div>
      <div id="selected-source-meta" class="muted"></div>
      <details>
        <summary>Advanced options</summary>
        <label>Input video path (manual override)
          <input id="input_path" placeholder="C:\\\\path\\\\to\\\\video.mp4" />
        </label>
      </details>
    </fieldset>

    <fieldset>
      <legend>2) Analyze</legend>
      <div class="chips">
        <button type="button" id="analyze-btn">Analyze selected video</button>
      </div>
      <div id="analysis-status" class="status-line">idle</div>
      <div id="analysis-summary" class="muted"></div>
    </fieldset>

    <fieldset>
      <legend>3) Validate preprocessing</legend>
      <h3>Recommended preprocessing</h3>
      <div id="recommendation-status" class="muted">Run analysis to get recommendations.</div>
      <table id="recommendations-table" class="hidden">
        <thead>
          <tr>
            <th>Operation</th>
            <th>Diagnostic</th>
            <th>Recommendation</th>
            <th>Confidence</th>
            <th>Explanation</th>
            <th>Validation / tweak command</th>
          </tr>
        </thead>
        <tbody id="recommendations-body"></tbody>
      </table>
      <h3>Crop configuration</h3>
      <label><input type="radio" name="crop_mode" value="preserve" checked /> Keep original frame</label>
      <label><input type="radio" name="crop_mode" value="auto_crop" /> Apply recommended crop (if reliable)</label>
      <label><input type="radio" name="crop_mode" value="custom_crop" /> Manual margins</label>
      <div class="row">
        <label>Top margin (px) <input id="crop_top" type="number" min="0" value="0" /></label>
        <label>Bottom margin (px) <input id="crop_bottom" type="number" min="0" value="0" /></label>
      </div>
      <div class="row">
        <label>Left margin (px) <input id="crop_left" type="number" min="0" value="0" /></label>
        <label>Right margin (px) <input id="crop_right" type="number" min="0" value="0" /></label>
      </div>
      <div id="crop-dimensions" class="muted"></div>
      <div id="preprocessing-plan-status" class="status-line">No plan generated yet.</div>
      <div class="muted">Unsupported remote operations are displayed as disabled (deinterlace/denoise/deblock in this phase).</div>
    </fieldset>

    <fieldset>
      <legend>4) Configure and launch upscale</legend>
      <label>Mode
        <select id="mode">
          <option value="automatic">Automatic (create + cleanup)</option>
          <option value="manual">Manual (existing instance, no auto-destroy)</option>
        </select>
      </label>
      <div class="row">
        <label>Target width <input id="target_width" type="number" min="1" /></label>
        <label>Target height <input id="target_height" type="number" min="1" /></label>
      </div>
      <label>Worker image tag <input id="worker_image" /></label>
      <label>GPU model <select id="gpu_model"></select></label>
      <label>Sort offers by
        <select id="sort_by">
          <option value="price">price</option>
          <option value="reliability">reliability</option>
          <option value="performance">performance</option>
        </select>
      </label>
      <label>Instance ID (manual mode required) <input id="instance_id" type="number" /></label>
      <details>
        <summary>Advanced options</summary>
        <label>Output path (optional) <input id="output_path" /></label>
        <label>Report path (optional) <input id="report_path" /></label>
        <label>Preprocessing plan path (manual override, optional) <input id="preprocessing_plan_path" /></label>
        <label>Offer ID (automatic mode optional) <input id="offer_id" type="number" /></label>
        <label>SSH bootstrap mode
          <select id="ssh_bootstrap_mode">
            <option value="tofu">tofu</option>
            <option value="strict">strict</option>
          </select>
        </label>
      </details>
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
    let latestAnalysisPayload = null;
    let latestPlanPayload = null;
    let analysisRequestToken = 0;
    let sourcesIndex = new Map();
    const handledPromptIds = new Set();
    const unsupportedRemoteOps = new Set(["deinterlace", "denoise", "deblock"]);

    function appendEvent(msg) {
      const box = document.getElementById("events");
      box.textContent += msg + "\\n";
      box.scrollTop = box.scrollHeight;
    }

    function prettyBytes(size) {
      if (typeof size !== "number" || Number.isNaN(size)) return "n/a";
      if (size < 1024) return `${size} B`;
      if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
      if (size < 1024 ** 3) return `${(size / (1024 ** 2)).toFixed(1)} MB`;
      return `${(size / (1024 ** 3)).toFixed(2)} GB`;
    }

    function readCropMode() {
      const selected = document.querySelector("input[name='crop_mode']:checked");
      return selected ? selected.value : "preserve";
    }

    function cropMarginsFromInputs() {
      return {
        top: Number(document.getElementById("crop_top").value || 0),
        bottom: Number(document.getElementById("crop_bottom").value || 0),
        left: Number(document.getElementById("crop_left").value || 0),
        right: Number(document.getElementById("crop_right").value || 0),
      };
    }

    function updateCropDimensionsPreview() {
      const info = document.getElementById("crop-dimensions");
      if (!latestAnalysisPayload || !latestAnalysisPayload.width || !latestAnalysisPayload.height) {
        info.textContent = "Resulting dimensions: n/a";
        return;
      }
      const srcW = Number(latestAnalysisPayload.width);
      const srcH = Number(latestAnalysisPayload.height);
      const mode = readCropMode();
      if (mode !== "custom_crop") {
        info.textContent = `Resulting dimensions: ${srcW} x ${srcH} (manual margins are ignored in this mode)`;
        return;
      }
      const m = cropMarginsFromInputs();
      const outW = srcW - m.left - m.right;
      const outH = srcH - m.top - m.bottom;
      info.textContent = `Resulting dimensions: ${outW} x ${outH} (source: ${srcW} x ${srcH})`;
    }

    function commandHintFor(opKey, planCropRect) {
      if (opKey === "crop") {
        if (!planCropRect) return "ffmpeg -i <input> -vf \\"crop=w:h:x:y\\" -f null -";
        return `ffmpeg -i <input> -vf "crop=${planCropRect.width}:${planCropRect.height}:${planCropRect.x}:${planCropRect.y}" -f null -`;
      }
      if (opKey === "deinterlace") {
        return "ffmpeg -i <input> -vf \\"idet,bwdif=mode=send_frame:parity=auto:deint=all\\" -f null -";
      }
      if (opKey === "denoise") {
        return "ffmpeg -i <input> -vf \\"hqdn3d=1.5:1.5:6:6\\" -f null -";
      }
      if (opKey === "deblock") {
        return "ffmpeg -i <input> -vf \\"deblock=filter=strong:block=4\\" -f null -";
      }
      return "n/a";
    }

    function diagnosticSummaryFor(opKey, analysis) {
      const visual = analysis?.analysis_report?.visual_diagnostics || {};
      if (opKey === "crop") {
        const crop = visual.cropdetect || {};
        const margins = crop.probable_margins;
        if (margins) {
          return `cropdetect: status=${crop.status || "unknown"}, margins T${margins.top || 0}/B${margins.bottom || 0}/L${margins.left || 0}/R${margins.right || 0}`;
        }
        return `cropdetect: status=${crop.status || "unknown"}`;
      }
      if (opKey === "deinterlace") {
        const idet = visual.idet || {};
        return `idet: classification=${idet.classification || "indeterminate"}, status=${idet.status || "unknown"}`;
      }
      if (opKey === "denoise" || opKey === "deblock") {
        const noise = visual.noise_artifacts || {};
        return `noise/artifacts: noise=${noise.likely_noise || "unknown"}, compression=${noise.likely_compression_artifacts || "unknown"}, status=${noise.status || "unknown"}`;
      }
      return "n/a";
    }

    function setRecommendationTableVisible(visible) {
      document.getElementById("recommendations-table").classList.toggle("hidden", !visible);
    }

    function renderRecommendations(plan, analysis) {
      const body = document.getElementById("recommendations-body");
      body.innerHTML = "";
      const byKey = new Map((plan.operations || []).map(op => [op.operation_id, op]));
      const cropOp = byKey.get("crop");
      const cropRect = cropOp?.parameters?.crop_rect || null;
      const labels = {
        crop: "Black-bar crop",
        deinterlace: "Deinterlace",
        denoise: "Noise reduction",
        deblock: "Compression artifact reduction",
      };

      for (const rec of plan.recommendations || []) {
        const op = byKey.get(rec.key);
        const tr = document.createElement("tr");
        const disabledNote = unsupportedRemoteOps.has(rec.key)
          ? "Disabled in this phase (worker does not execute it remotely)."
          : "Executable remotely.";
        tr.innerHTML = `
          <td><strong>${labels[rec.key] || rec.key}</strong><div class="muted">${disabledNote}</div></td>
          <td>${diagnosticSummaryFor(rec.key, analysis)}</td>
          <td>${rec.recommended_action || "unknown"}</td>
          <td>${rec.confidence || "unknown"}</td>
          <td>${rec.reason || ""}</td>
          <td><code>${commandHintFor(rec.key, cropRect)}</code></td>
        `;
        body.appendChild(tr);
      }
      setRecommendationTableVisible((plan.recommendations || []).length > 0);
      const warnings = (plan.warnings || []).join(" | ");
      document.getElementById("recommendation-status").textContent = warnings
        ? `Recommendations ready. Warnings: ${warnings}`
        : "Recommendations ready.";
    }

    function currentUserChoices() {
      const mode = readCropMode();
      const choices = {
        crop_mode: mode,
        custom_crop_margins: null,
        deinterlace_mode: "preserve",
        denoise_mode: "preserve",
        deblock_mode: "preserve",
      };
      if (mode === "custom_crop") {
        choices.custom_crop_margins = cropMarginsFromInputs();
      }
      return choices;
    }

    async function loadVideoSources() {
      const res = await fetch("/api/video/sources");
      const data = await res.json();
      if (!res.ok) {
        appendEvent("Failed to list video sources: " + (data.detail || JSON.stringify(data)));
        return;
      }
      const select = document.getElementById("source_name");
      select.innerHTML = '<option value="">-- Select a video --</option>';
      sourcesIndex = new Map();
      for (const item of data.sources || []) {
        const opt = document.createElement("option");
        opt.value = item.name;
        const modified = item.modified_at ? new Date(item.modified_at).toLocaleString() : "n/a";
        opt.textContent = `${item.name} (${prettyBytes(item.size_bytes)}, ${modified})`;
        select.appendChild(opt);
        sourcesIndex.set(item.name, item);
      }
      document.getElementById("video-sources-dir").textContent = `Sources directory: ${data.directory}`;
      appendEvent(`Video sources refreshed: ${(data.sources || []).length} selectable file(s).`);
    }

    async function rebuildPlanFromCurrentAnalysis(token) {
      if (!latestAnalysisPayload) return;
      const planStatus = document.getElementById("preprocessing-plan-status");
      planStatus.textContent = "Building preprocessing plan...";
      const response = await fetch("/api/video/preprocessing-plan", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          analysis_payload: latestAnalysisPayload,
          user_choices: currentUserChoices(),
        }),
      });
      const payload = await response.json();
      if (token !== analysisRequestToken) return;
      if (!response.ok) {
        latestPlanPayload = null;
        planStatus.textContent = "Preprocessing plan error: " + (payload.detail || JSON.stringify(payload));
        setRecommendationTableVisible(false);
        return;
      }
      latestPlanPayload = payload;
      renderRecommendations(payload, latestAnalysisPayload);
      const enabled = (payload.operations || []).filter(op => op.enabled).map(op => op.operation_id);
      planStatus.textContent = enabled.length
        ? `Plan ready. Enabled operations: ${enabled.join(", ")}`
        : "Plan ready. No operation is enabled (legacy behavior on launch).";
    }

    async function analyzeSelectedSource() {
      const sourceName = document.getElementById("source_name").value;
      if (!sourceName) {
        document.getElementById("analysis-status").textContent = "No source selected.";
        return;
      }
      const token = ++analysisRequestToken;
      latestAnalysisPayload = null;
      latestPlanPayload = null;
      setRecommendationTableVisible(false);
      document.getElementById("analysis-status").textContent = "Analysis running...";
      document.getElementById("analysis-summary").textContent = "";
      document.getElementById("recommendation-status").textContent = "Waiting for analysis...";
      const response = await fetch("/api/video/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          source_name: sourceName,
          include_visual_diagnostics: true,
        }),
      });
      const payload = await response.json();
      if (token !== analysisRequestToken) return;
      if (!response.ok) {
        document.getElementById("analysis-status").textContent = "Analysis failed.";
        document.getElementById("analysis-summary").textContent = payload.detail || JSON.stringify(payload);
        return;
      }
      latestAnalysisPayload = payload;
      const dims = payload.width && payload.height ? `${payload.width} x ${payload.height}` : "unknown size";
      document.getElementById("analysis-status").textContent = "Analysis completed.";
      document.getElementById("analysis-summary").textContent = `Codec: ${payload.video_codec || "unknown"}, ${dims}, fps: ${payload.frame_rate_fps || "unknown"}`;
      updateCropDimensionsPreview();
      await rebuildPlanFromCurrentAnalysis(token);
    }

    function selectedSourceChanged() {
      const selected = document.getElementById("source_name").value;
      const meta = sourcesIndex.get(selected);
      if (!selected || !meta) {
        document.getElementById("selected-source-meta").textContent = "";
        return;
      }
      const modified = meta.modified_at ? new Date(meta.modified_at).toLocaleString() : "n/a";
      document.getElementById("selected-source-meta").textContent = `Selected: ${meta.name} | ${prettyBytes(meta.size_bytes)} | modified ${modified}`;
      analyzeSelectedSource();
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
      const sourceName = document.getElementById("source_name").value || null;
      const manualInput = document.getElementById("input_path").value.trim();
      if (!sourceName && !manualInput) {
        appendEvent("Start failed: choose a source from video/ or provide a manual input path in advanced options.");
        return;
      }
      const manualPlanPath = document.getElementById("preprocessing_plan_path").value.trim() || null;
      const enabledRemoteCapable = (latestPlanPayload?.operations || []).some(
        (op) => op.enabled && !unsupportedRemoteOps.has(op.operation_id)
      );
      const payload = {
        mode: document.getElementById("mode").value,
        source_name: sourceName,
        input_path: sourceName ? null : manualInput,
        output_path: document.getElementById("output_path").value || null,
        report_path: document.getElementById("report_path").value || null,
        preprocessing_plan_path: manualPlanPath,
        preprocessing_plan_payload: manualPlanPath ? null : (enabledRemoteCapable ? latestPlanPayload : null),
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
      if (manualPlanPath && latestPlanPayload) {
        appendEvent("Manual preprocessing plan path is set: it overrides UI-generated preprocessing choices.");
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

    document.getElementById("refresh-sources-btn").addEventListener("click", loadVideoSources);
    document.getElementById("source_name").addEventListener("change", selectedSourceChanged);
    document.getElementById("analyze-btn").addEventListener("click", analyzeSelectedSource);
    for (const radio of document.querySelectorAll("input[name='crop_mode']")) {
      radio.addEventListener("change", async () => {
        updateCropDimensionsPreview();
        const token = ++analysisRequestToken;
        await rebuildPlanFromCurrentAnalysis(token);
      });
    }
    for (const id of ["crop_top", "crop_bottom", "crop_left", "crop_right"]) {
      document.getElementById(id).addEventListener("input", async () => {
        updateCropDimensionsPreview();
        if (readCropMode() !== "custom_crop") return;
        const token = ++analysisRequestToken;
        await rebuildPlanFromCurrentAnalysis(token);
      });
    }

    loadDefaults();
    loadVideoSources();
  </script>
</body>
</html>
"""
