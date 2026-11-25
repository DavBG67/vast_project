from __future__ import annotations

import json
import re
import subprocess
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
from vast_upscaler.remote_seedvr2_worker import (
    RemoteSeedvr2WorkerPreparationError,
    RemoteSeedvr2WorkerProcessError,
    RemoteSeedvr2WorkerRecoveryError,
    Seedvr2RemoteRequest,
    execute_remote_seedvr2_worker,
    recover_remote_seedvr2_results,
)
from vast_upscaler.remote_worker import (
    RemoteWorkerRequest,
    execute_remote_worker,
)
from vast_upscaler.report_enrichment import enrich_local_report_with_vast_metadata
from vast_upscaler.session_store import SessionStore
from vast_upscaler.source_analysis import (
    SourceAnalysisError,
    VisualDiagnosticsOptions,
    analyze_source_video,
    analyze_source_video_report,
)
from vast_upscaler.transfer import (
    OpenSshTransport,
    SshConnectionInfo,
    SshTransportError,
)
from vast_upscaler.vast_client import VastClient, VastClientConfig
from vast_upscaler.vast_write_client import InstanceSummary, VastWriteClient
from vast_upscaler.video import VideoProbeError, inspect_video
from vast_upscaler.video_excerpt import (
    VideoExcerptError,
    VideoExcerptResult,
    create_video_excerpt,
    resolve_excerpt_window,
)

RunMode = Literal["automatic", "manual"]
SortBy = Literal["price", "reliability", "performance"]
SshBootstrapMode = Literal["tofu", "strict"]
EngineName = Literal["realesrgan", "seedvr2"]
Seedvr2Profile = Literal["smoke", "reference"]

_DEFAULT_WORKER_IMAGE = ""
_WORKER_IMAGE_HISTORY_PATH = Path.home() / ".vast_upscaler" / "web_worker_images.json"
_SEEDVR2_RECOVERY_STATE_PATH = Path.home() / ".vast_upscaler" / "seedvr2_recovery_jobs.json"
_DEFAULT_ENGINE: EngineName = "realesrgan"
_SUPPORTED_ENGINES: tuple[EngineName, ...] = ("realesrgan", "seedvr2")
_SEEDVR2_PROFILES: tuple[Seedvr2Profile, ...] = ("smoke", "reference")
_WORKER_IMAGE_REPO_BY_ENGINE: dict[EngineName, str] = {
    "realesrgan": "ghcr.io/davbg67/vast-video-upscaler-worker",
    "seedvr2": "ghcr.io/davbg67/vast-video-upscaler-seedvr2",
}
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
_ALLOWED_OUTPUT_CONTAINER_EXTENSIONS = {".mp4"}


class WebWorkflowCancelled(RuntimeError):
    pass


class StartJobRequest(BaseModel):
    mode: RunMode
    engine: EngineName = _DEFAULT_ENGINE
    seedvr2_profile: Seedvr2Profile = "smoke"
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
    excerpt_enabled: bool = False
    excerpt_start: str | None = None
    excerpt_duration_seconds: float = Field(default=10.0, gt=0)
    recover_seedvr2_job_id: str | None = None


class PromptReplyRequest(BaseModel):
    approved: bool | None = None
    fingerprint: str | None = None


class Seedvr2RecoveryRequest(BaseModel):
    recovery_job_id: str
    instance_id: int | None = None


class SourceAnalysisRequest(BaseModel):
    input_path: str | None = None
    source_name: str | None = None
    include_visual_diagnostics: bool = False
    diagnostic_options: VisualDiagnosticsOptions | None = None
    excerpt_enabled: bool = False
    excerpt_start: str | None = None
    excerpt_duration_seconds: float = Field(default=10.0, gt=0)


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


@dataclass(frozen=True)
class _EffectiveJobInput:
    input_path: Path
    original_source: Path
    excerpt_result: VideoExcerptResult | None
    cleanup_excerpt_path: Path | None


class _Job:
    def __init__(self, *, request: StartJobRequest, defaults: dict[str, Any]) -> None:
        self.job_id = f"job-{uuid4().hex[:12]}"
        self.request = request
        self.defaults = defaults
        self.created_at = datetime.now(UTC).isoformat()
        self.updated_at = self.created_at
        self.status = "pending"
        self.business_state = "pending"
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
        latest_worker_image = _load_latest_known_worker_image(engine=_DEFAULT_ENGINE)
        latest_worker_image_seedvr2 = _load_latest_known_worker_image(engine="seedvr2")
        return {
            "default_engine": _DEFAULT_ENGINE,
            "engines": list(_SUPPORTED_ENGINES),
            "seedvr2_profiles": list(_SEEDVR2_PROFILES),
            "target_width": settings.video.target_width,
            "target_height": settings.video.target_height,
            "gpu_models": settings.vast.gpu_models,
            "worker_image_latest_known": latest_worker_image,
            "worker_image_latest_by_engine": {
                _DEFAULT_ENGINE: latest_worker_image,
                "seedvr2": latest_worker_image_seedvr2,
            },
            "worker_image_source": "history" if latest_worker_image else "none",
            "worker_image_required": latest_worker_image == "",
            "ssh_bootstrap_mode": "tofu",
            "seedvr2_recovery_jobs": _list_seedvr2_recovery_jobs(),
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
                "engine": _normalize_engine(request.engine),
                "seedvr2_profile": _normalize_seedvr2_profile(request.seedvr2_profile),
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
                "excerpt_enabled": request.excerpt_enabled,
                "excerpt_start": request.excerpt_start,
                "excerpt_duration_seconds": request.excerpt_duration_seconds,
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
        job.business_state = "running"
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
            job.business_state = "completed"
            if job.request.worker_image:
                try:
                    _store_latest_worker_image(job.request.worker_image, engine=_normalize_engine(job.request.engine))
                except (HTTPException, OSError, TypeError, ValueError) as exc:
                    job.append_event(
                        "status",
                        {
                            "state": "warning",
                            "message": f"Could not persist latest worker image suggestion: {exc}",
                        },
                    )
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
            job.business_state = "cancelled"
            job.error = "Cancelled by operator."
            job.append_event("error", {"message": job.error})
            if job.request.mode == "manual":
                job.append_event(
                    "instance",
                    {"state": "manual-instance-preserved", "message": "Manual mode never destroys the existing instance automatically."},
                )
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            if isinstance(exc, RemoteSeedvr2WorkerRecoveryError):
                job.business_state = "failed_recovery"
            elif isinstance(exc, RemoteSeedvr2WorkerProcessError):
                job.business_state = "failed_inference"
            elif isinstance(exc, RemoteSeedvr2WorkerPreparationError):
                job.business_state = "failed_preparation"
            else:
                job.business_state = "failed"
            job.error = str(exc)
            job.append_event("error", {"message": str(exc)})
            if job.request.mode == "manual":
                job.append_event(
                    "instance",
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
        kind = str(payload.get("kind") or "")
        stage = str(payload.get("stage") or "")
        status = str(payload.get("status") or "")
        if kind == "transfer" and stage == "download" and status == "started":
            job.business_state = "inference_succeeded_recovering"
            job.append_event("status", {"state": "inference_succeeded_recovering", "message": "Inference finished, downloading artifacts."})
        elif stage and job.business_state != "inference_succeeded_recovering":
            job.business_state = "running"
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
            "business_state": job.business_state,
            "mode": job.request.mode,
            "engine": _normalize_engine(job.request.engine),
            "seedvr2_profile": _normalize_seedvr2_profile(job.request.seedvr2_profile),
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
        recovery_job_id = (request.recover_seedvr2_job_id or "").strip()
        if recovery_job_id:
            engine = _normalize_engine(request.engine)
            if engine != "seedvr2" or request.mode != "manual":
                raise HTTPException(status_code=400, detail="SeedVR2 recovery requires manual mode and engine=seedvr2.")
            _normalize_seedvr2_profile(request.seedvr2_profile)
            return

        source_name = (request.source_name or "").strip()
        input_path_raw = (request.input_path or "").strip()
        if source_name and input_path_raw:
            raise HTTPException(status_code=400, detail="Provide either source_name or input_path, not both.")
        if not source_name and not input_path_raw:
            raise HTTPException(status_code=400, detail="Either source_name or input_path is required.")

        input_path = _resolve_input_video_path(request=request)
        if not input_path.exists() or not input_path.is_file():
            raise HTTPException(status_code=400, detail="Input path must be an existing file.")
        _validate_excerpt_request(
            enabled=request.excerpt_enabled,
            start_timecode=request.excerpt_start,
            duration_seconds=request.excerpt_duration_seconds,
            source_path=input_path,
        )
        output_path = _resolve_output_path(request=request)
        _validate_output_container_compatibility(output_path=output_path)
        report_path = _resolve_report_path(request=request, output_path=output_path)
        _validate_local_decode_capability(input_path=input_path)
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
            if _normalize_engine(request.engine) == "seedvr2":
                raise HTTPException(status_code=400, detail="SeedVR2 is available in manual mode only.")
            image = (request.worker_image or _load_latest_known_worker_image(engine=_normalize_engine(request.engine))).strip()
            if not image:
                raise HTTPException(
                    status_code=400,
                    detail="Automatic mode requires an explicit worker image tag (no placeholder/default available).",
                )
            if image.lower() in _WORKER_IMAGE_PLACEHOLDERS:
                raise HTTPException(status_code=400, detail=f"Refusing placeholder worker image tag: {image}")
        engine = _normalize_engine(request.engine)
        _normalize_seedvr2_profile(request.seedvr2_profile)
        if engine == "seedvr2" and request.mode != "manual":
            raise HTTPException(status_code=400, detail="SeedVR2 is available in manual mode only.")
        if request.worker_image:
            _ensure_image_repository_matches_engine(image=request.worker_image, engine=engine)


def _default_auto_runner(request: StartJobRequest, settings: Settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
    callbacks.raise_if_cancelled()
    engine = _normalize_engine(request.engine)
    if engine != "realesrgan":
        raise OrchestrationError("Automatic mode currently supports Real-ESRGAN only.")
    effective_settings = settings.model_copy(deep=True)
    if request.gpu_model:
        effective_settings.vast.gpu_models = [request.gpu_model]

    output_path = _resolve_output_path(request=request)
    report_path = _resolve_report_path(request=request, output_path=output_path)
    source_path = _resolve_input_video_path(request=request)
    effective_input = _prepare_effective_job_input(request=request, source_path=source_path, job_prefix="job-excerpt")
    input_path = effective_input.input_path
    preprocessing_contract = _resolve_remote_preprocessing_contract(request=request)
    worker_image = (request.worker_image or _load_latest_known_worker_image(engine=engine)).strip()
    if not worker_image:
        raise OrchestrationError("Worker image tag is required.")
    if worker_image.lower() in _WORKER_IMAGE_PLACEHOLDERS:
        raise OrchestrationError(f"Refusing placeholder worker image tag: {worker_image}")
    _ensure_image_repository_matches_engine(image=worker_image, engine=engine)
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
            final_report_path = (
                summary.worker_result.local_report_path if summary.worker_result is not None else report_path
            )
            if effective_input is not None:
                _safe_merge_report_metadata(
                    report_path=final_report_path,
                    metadata=_build_excerpt_report_metadata(effective_input=effective_input),
                )
            return WorkflowExecutionResult(
                mode="automatic",
                session_id=summary.session_id,
                state=summary.state,
                output_path=str(summary.worker_result.local_output_path) if summary.worker_result is not None else None,
                report_path=str(final_report_path),
                duration_seconds=0.0,
                estimated_total_usd=summary.estimate.total_estimated_usd,
                instance_id=summary.instance_id,
                instance_state=summary.state,
                cleanup_verified=summary.cleanup_verified,
                cleanup_warning=summary.cleanup_warning,
            )
        except Exception:
            if effective_input is not None:
                _safe_merge_report_metadata(
                    report_path=report_path,
                    metadata=_build_excerpt_report_metadata(effective_input=effective_input),
                )
            raise
    finally:
        _cleanup_job_excerpt(effective_input.cleanup_excerpt_path)
        read_client.close()
        write_client.close()


def _default_manual_runner(request: StartJobRequest, settings: Settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
    callbacks.raise_if_cancelled()
    recovery_job_id = (request.recover_seedvr2_job_id or "").strip() or None
    if request.instance_id is None and recovery_job_id is None:
        raise OrchestrationError("Manual mode requires an existing Vast instance id.")
    engine = _normalize_engine(request.engine)
    if recovery_job_id and engine != "seedvr2":
        raise OrchestrationError("Recovery-only mode is available for SeedVR2 manual jobs only.")
    seedvr2_profile = _normalize_seedvr2_profile(request.seedvr2_profile)
    effective_worker_image = (request.worker_image or _load_latest_known_worker_image(engine=engine)).strip() or None
    if effective_worker_image:
        _ensure_image_repository_matches_engine(image=effective_worker_image, engine=engine)

    effective_input: _EffectiveJobInput | None = None
    preprocessing_contract = None
    if recovery_job_id:
        recovery_record = _load_seedvr2_recovery_job(recovery_job_id)
        output_path = Path(str(recovery_record["local_output_path"]))
        report_path = Path(str(recovery_record["local_report_path"]))
        input_path = Path(str(recovery_record.get("local_input_path") or output_path))
        if request.instance_id is None:
            request.instance_id = int(recovery_record["instance_id"])
    else:
        output_path = _resolve_output_path(request=request)
        report_path = _resolve_report_path(request=request, output_path=output_path)
        source_path = _resolve_input_video_path(request=request)
        effective_input = _prepare_effective_job_input(request=request, source_path=source_path, job_prefix="job-excerpt")
        input_path = effective_input.input_path
        preprocessing_contract = _resolve_remote_preprocessing_contract(request=request)

    transport = OpenSshTransport()
    write_client = _build_write_client(settings)
    started = time.perf_counter()
    instance: InstanceSummary | None = None
    worker_image_identity: dict[str, Any] = {
        "requested": (request.worker_image or "").strip() or None,
        "suggested_or_expected": effective_worker_image,
        "instance_declared": None,
        "expected_repository": _WORKER_IMAGE_REPO_BY_ENGINE[engine],
        "verification_level": "not-checked",
    }
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
                    worker_image=effective_worker_image or "",
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
        _run_manual_engine_preflight(
            transport=transport,
            connection=connection,
            engine=engine,
        )
        worker_image_identity = _check_manual_instance_image_compatibility(
            instance=instance,
            requested_image=request.worker_image,
            effective_worker_image=effective_worker_image,
            engine=engine,
            callbacks=callbacks,
        )
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
        if engine == "seedvr2":
            if recovery_job_id:
                callbacks.report_worker_progress({"kind": "worker", "stage": "artifact-recovery", "status": "started", "determinate": False})
                remote_root_base = str(recovery_record.get("remote_root_base") or "/workspace/seedvr2-sessions")
                remote_session_id = str(recovery_record.get("remote_session_id") or recovery_job_id)
                worker_result = recover_remote_seedvr2_results(
                    transport=transport,
                    connection=connection,
                    request=Seedvr2RemoteRequest(
                        session_id=remote_session_id,
                        local_input_path=input_path,
                        local_output_path=output_path,
                        local_report_path=report_path,
                        profile=seedvr2_profile,
                        remote_root_base=remote_root_base,
                    ),
                    progress_callback=lambda payload: callbacks.report_worker_progress(payload),
                )
                _update_seedvr2_recovery_job_state(
                    recovery_job_id=recovery_job_id,
                    inference_state="completed",
                    recovery_state="completed",
                )
            else:
                callbacks.report_worker_progress(
                    {
                        "kind": "worker",
                        "stage": "seedvr2-prepared",
                        "progress": 0.05,
                        "message": f"SeedVR2 profile '{seedvr2_profile}' selected (input excerpt prepared locally).",
                    }
                )
                remote_session_id = f"manual-{uuid4().hex[:12]}"
                _upsert_seedvr2_recovery_job(
                    {
                        "recovery_job_id": remote_session_id,
                        "created_at": datetime.now(UTC).isoformat(),
                        "updated_at": datetime.now(UTC).isoformat(),
                        "instance_id": instance.instance_id,
                        "engine": "seedvr2",
                        "seedvr2_profile": seedvr2_profile,
                        "remote_root_base": "/workspace/seedvr2-sessions",
                        "remote_session_id": remote_session_id,
                        "ssh_host": instance.ssh_host,
                        "ssh_port": instance.ssh_port,
                        "local_input_path": str(input_path),
                        "local_output_path": str(output_path),
                        "local_report_path": str(report_path),
                        "inference_state": "running",
                        "recovery_state": "pending",
                    }
                )
                worker_result = execute_remote_seedvr2_worker(
                    transport=transport,
                    connection=connection,
                    request=Seedvr2RemoteRequest(
                        session_id=remote_session_id,
                        local_input_path=input_path,
                        local_output_path=output_path,
                        local_report_path=report_path,
                        profile=seedvr2_profile,
                    ),
                    progress_callback=lambda payload: callbacks.report_worker_progress(payload),
                )
                _update_seedvr2_recovery_job_state(
                    recovery_job_id=remote_session_id,
                    inference_state="completed",
                    recovery_state="completed",
                )
            final_report_path = worker_result.local_report_path
        else:
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
            final_report_path = worker_result.local_report_path
        enrich_local_report_with_vast_metadata(
            report_path=final_report_path,
            mode="manual",
            requested_worker_image=request.worker_image,
            instance=instance,
            offer=None,
        )
        _safe_merge_report_metadata(
            report_path=final_report_path,
            metadata={
                "engine": {
                    "requested": request.engine,
                    "effective": engine,
                    "seedvr2_profile": seedvr2_profile if engine == "seedvr2" else None,
                },
                "worker_image": {
                    "requested": worker_image_identity["requested"],
                    "suggested_or_expected": worker_image_identity["suggested_or_expected"],
                    "instance_declared": worker_image_identity["instance_declared"],
                    "expected_repository": worker_image_identity["expected_repository"],
                    "verification_level": worker_image_identity["verification_level"],
                },
            },
        )
        if effective_input is not None:
            _safe_merge_report_metadata(
                report_path=final_report_path,
                metadata=_build_excerpt_report_metadata(effective_input=effective_input),
            )
        elapsed = time.perf_counter() - started
        return WorkflowExecutionResult(
            mode="manual",
            session_id=None,
            state="COMPLETED",
            output_path=str(worker_result.local_output_path),
            report_path=str(final_report_path),
            duration_seconds=elapsed,
            estimated_total_usd=None,
            instance_id=instance.instance_id,
            instance_state=instance.actual_status,
            cleanup_verified=False,
            cleanup_warning="Manual mode leaves instance running. Destroy manually when done.",
        )
    except RemoteSeedvr2WorkerRecoveryError:
        if recovery_job_id and engine == "seedvr2":
            _update_seedvr2_recovery_job_state(
                recovery_job_id=recovery_job_id,
                inference_state="completed",
                recovery_state="failed",
            )
        elif engine == "seedvr2":
            _update_seedvr2_recovery_job_state(
                recovery_job_id=remote_session_id,  # type: ignore[name-defined]
                inference_state="completed",
                recovery_state="failed",
            )
        raise
    except RemoteSeedvr2WorkerPreparationError:
        if recovery_job_id and engine == "seedvr2":
            _update_seedvr2_recovery_job_state(
                recovery_job_id=recovery_job_id,
                inference_state="failed",
                recovery_state="pending",
            )
        elif engine == "seedvr2":
            _update_seedvr2_recovery_job_state(
                recovery_job_id=remote_session_id,  # type: ignore[name-defined]
                inference_state="failed",
                recovery_state="pending",
            )
        raise
    except RemoteSeedvr2WorkerProcessError:
        if recovery_job_id and engine == "seedvr2":
            _update_seedvr2_recovery_job_state(
                recovery_job_id=recovery_job_id,
                inference_state="failed",
                recovery_state="pending",
            )
        elif engine == "seedvr2":
            _update_seedvr2_recovery_job_state(
                recovery_job_id=remote_session_id,  # type: ignore[name-defined]
                inference_state="failed",
                recovery_state="pending",
            )
        raise
    except Exception:
        enrich_local_report_with_vast_metadata(
            report_path=report_path,
            mode="manual",
            requested_worker_image=request.worker_image,
            instance=instance,
            offer=None,
        )
        _safe_merge_report_metadata(
            report_path=report_path,
            metadata={
                "engine": {
                    "requested": request.engine,
                    "effective": engine,
                    "seedvr2_profile": seedvr2_profile if engine == "seedvr2" else None,
                },
                "worker_image": {
                    "requested": worker_image_identity["requested"],
                    "suggested_or_expected": worker_image_identity["suggested_or_expected"],
                    "instance_declared": worker_image_identity["instance_declared"],
                    "expected_repository": worker_image_identity["expected_repository"],
                    "verification_level": worker_image_identity["verification_level"],
                },
            },
        )
        if effective_input is not None:
            _safe_merge_report_metadata(
                report_path=report_path,
                metadata=_build_excerpt_report_metadata(effective_input=effective_input),
            )
        raise
    finally:
        if effective_input is not None:
            _cleanup_job_excerpt(effective_input.cleanup_excerpt_path)
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
    return source.with_name(f"{source.stem}-upscaled.mp4")


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


def _excerpts_dir() -> Path:
    return _project_root_path() / "data" / "excerpts"


def _cleanup_job_excerpt(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        return


def _safe_merge_report_metadata(*, report_path: Path, metadata: dict[str, Any]) -> None:
    if not report_path.exists() or not report_path.is_file():
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload.update(metadata)
    try:
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def _build_excerpt_report_metadata(*, effective_input: _EffectiveJobInput) -> dict[str, Any]:
    excerpt = effective_input.excerpt_result
    return {
        "original_source": str(effective_input.original_source),
        "excerpt_enabled": excerpt is not None,
        "excerpt_start_seconds": excerpt.start_seconds if excerpt is not None else None,
        "excerpt_requested_duration_seconds": excerpt.requested_duration_seconds if excerpt is not None else None,
        "excerpt_actual_duration_seconds": excerpt.actual_duration_seconds if excerpt is not None else None,
        "excerpt_source": str(excerpt.excerpt_path) if excerpt is not None else None,
    }


def _prepare_effective_job_input(*, request: StartJobRequest, source_path: Path, job_prefix: str) -> _EffectiveJobInput:
    if not request.excerpt_enabled:
        return _EffectiveJobInput(
            input_path=source_path,
            original_source=source_path,
            excerpt_result=None,
            cleanup_excerpt_path=None,
        )

    excerpt_dir = _excerpts_dir()
    excerpt_file = excerpt_dir / f"{job_prefix}-{uuid4().hex[:16]}.mkv"
    excerpt_result = _create_excerpt_for_range(
        source_path=source_path,
        excerpt_path=excerpt_file,
        start_timecode=request.excerpt_start,
        requested_duration_seconds=request.excerpt_duration_seconds,
    )
    return _EffectiveJobInput(
        input_path=excerpt_result.excerpt_path,
        original_source=source_path,
        excerpt_result=excerpt_result,
        cleanup_excerpt_path=excerpt_result.excerpt_path,
    )


def _create_excerpt_for_range(
    *,
    source_path: Path,
    excerpt_path: Path,
    start_timecode: str | None,
    requested_duration_seconds: float | None,
) -> VideoExcerptResult:
    try:
        source_info = inspect_video(source_path)
    except VideoProbeError as exc:
        raise HTTPException(status_code=400, detail=f"Could not inspect source for excerpt validation: {exc}") from exc
    try:
        start_seconds, duration_seconds = resolve_excerpt_window(
            start_timecode=start_timecode,
            requested_duration_seconds=requested_duration_seconds,
            source_duration_seconds=source_info.duration_seconds,
        )
        return create_video_excerpt(
            source_path=source_path,
            excerpt_path=excerpt_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
    except VideoExcerptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_excerpt_request(*, enabled: bool, start_timecode: str | None, duration_seconds: float | None, source_path: Path) -> None:
    if not enabled:
        return
    try:
        source_info = inspect_video(source_path)
    except VideoProbeError as exc:
        raise HTTPException(status_code=400, detail=f"Could not inspect source for excerpt validation: {exc}") from exc
    try:
        resolve_excerpt_window(
            start_timecode=start_timecode,
            requested_duration_seconds=duration_seconds,
            source_duration_seconds=source_info.duration_seconds,
        )
    except VideoExcerptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_output_container_compatibility(*, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix not in _ALLOWED_OUTPUT_CONTAINER_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_OUTPUT_CONTAINER_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported output container '{output_path.suffix or '<none>'}'. "
                f"This phase supports MP4/H.264/AAC output only. Allowed extensions: {allowed}."
            ),
        )


def _validate_local_decode_capability(*, input_path: Path, ffmpeg_bin: str = "ffmpeg") -> None:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        run = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Local ffmpeg is unavailable for codec validation: {exc}") from exc
    if run.returncode != 0:
        message = (run.stderr or "").strip() or "Unsupported or undecodable source video stream."
        raise HTTPException(status_code=400, detail=f"Source decode precheck failed: {message}")


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


def _preview_dimensions_from_plan_payload(plan_payload: dict[str, Any]) -> dict[str, int] | None:
    source_summary = plan_payload.get("source_summary")
    if not isinstance(source_summary, dict):
        return None
    width = source_summary.get("width")
    height = source_summary.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return None

    operations = plan_payload.get("operations")
    if not isinstance(operations, list):
        return {"width": width, "height": height}
    crop_operation = next(
        (
            item
            for item in operations
            if isinstance(item, dict) and item.get("operation_id") == "crop" and item.get("enabled") is True
        ),
        None,
    )
    if not isinstance(crop_operation, dict):
        return {"width": width, "height": height}
    parameters = crop_operation.get("parameters")
    if not isinstance(parameters, dict):
        return {"width": width, "height": height}
    crop_rect = parameters.get("crop_rect")
    if not isinstance(crop_rect, dict):
        return {"width": width, "height": height}
    out_width = crop_rect.get("width")
    out_height = crop_rect.get("height")
    if not isinstance(out_width, int) or not isinstance(out_height, int) or out_width <= 0 or out_height <= 0:
        return {"width": width, "height": height}
    return {"width": out_width, "height": out_height}


def _normalize_engine(value: str | None) -> EngineName:
    candidate = (value or _DEFAULT_ENGINE).strip().lower()
    if candidate in _SUPPORTED_ENGINES:
        return candidate  # type: ignore[return-value]
    raise HTTPException(status_code=400, detail=f"Unsupported engine: {value}")


def _normalize_seedvr2_profile(value: str | None) -> Seedvr2Profile:
    candidate = (value or "smoke").strip().lower()
    if candidate in _SEEDVR2_PROFILES:
        return candidate  # type: ignore[return-value]
    raise HTTPException(status_code=400, detail=f"Unsupported SeedVR2 profile: {value}")


def _parse_image_repository(image: str) -> str | None:
    text = image.strip()
    if not text or text.lower() in _WORKER_IMAGE_PLACEHOLDERS:
        return None
    repository = text.split("@", 1)[0].split(":", 1)[0].strip()
    if not repository:
        return None
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", repository):
        return None
    return repository.lower()


def _ensure_image_repository_matches_engine(*, image: str, engine: EngineName) -> None:
    parsed_repo = _parse_image_repository(image)
    if parsed_repo is None:
        raise HTTPException(status_code=400, detail=f"Invalid worker image reference: {image}")
    expected_repo = _WORKER_IMAGE_REPO_BY_ENGINE[engine]
    if parsed_repo != expected_repo:
        raise HTTPException(
            status_code=400,
            detail=f"Worker image repository mismatch for engine '{engine}': expected {expected_repo}, got {parsed_repo}.",
        )


def _read_worker_image_history_payload() -> dict[str, Any]:
    try:
        payload = json.loads(_WORKER_IMAGE_HISTORY_PATH.read_text(encoding="utf-8"))
    except OSError:
        return {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _load_latest_known_worker_image(*, engine: EngineName = _DEFAULT_ENGINE) -> str:
    payload = _read_worker_image_history_payload()
    expected_repo = _WORKER_IMAGE_REPO_BY_ENGINE[engine]

    def _normalized_if_compatible(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized or normalized.lower() in _WORKER_IMAGE_PLACEHOLDERS:
            return None
        repository = _parse_image_repository(normalized)
        if repository != expected_repo:
            return None
        return normalized

    engines = payload.get("engines")
    if isinstance(engines, dict):
        entry = engines.get(engine)
        if isinstance(entry, dict):
            compatible = _normalized_if_compatible(entry.get("latest"))
            if compatible is not None:
                return compatible
    if engine == _DEFAULT_ENGINE:
        compatible = _normalized_if_compatible(payload.get("latest"))
        if compatible is not None:
            return compatible
    return _DEFAULT_WORKER_IMAGE


def _store_latest_worker_image(value: str, *, engine: EngineName = _DEFAULT_ENGINE) -> None:
    image = value.strip()
    if not image:
        return
    if image.lower() in _WORKER_IMAGE_PLACEHOLDERS:
        return
    _ensure_image_repository_matches_engine(image=image, engine=engine)
    payload = _read_worker_image_history_payload()
    engines = payload.get("engines")
    if not isinstance(engines, dict):
        engines = {}
    engine_entry = engines.get(engine)
    if not isinstance(engine_entry, dict):
        engine_entry = {}
    engine_entry["latest"] = image
    engines[engine] = engine_entry
    payload["engines"] = engines
    if engine == _DEFAULT_ENGINE:
        payload["latest"] = image
    _WORKER_IMAGE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WORKER_IMAGE_HISTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_seedvr2_recovery_payload() -> dict[str, Any]:
    try:
        payload = json.loads(_SEEDVR2_RECOVERY_STATE_PATH.read_text(encoding="utf-8"))
    except OSError:
        return {"jobs": {}}
    except json.JSONDecodeError:
        return {"jobs": {}}
    if not isinstance(payload, dict):
        return {"jobs": {}}
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        payload["jobs"] = {}
    return payload


def _write_seedvr2_recovery_payload(payload: dict[str, Any]) -> None:
    _SEEDVR2_RECOVERY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SEEDVR2_RECOVERY_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _upsert_seedvr2_recovery_job(job_payload: dict[str, Any]) -> None:
    payload = _read_seedvr2_recovery_payload()
    jobs = payload.setdefault("jobs", {})
    assert isinstance(jobs, dict)
    jobs[job_payload["recovery_job_id"]] = job_payload
    _write_seedvr2_recovery_payload(payload)


def _update_seedvr2_recovery_job_state(*, recovery_job_id: str, inference_state: str | None = None, recovery_state: str | None = None) -> None:
    payload = _read_seedvr2_recovery_payload()
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        return
    record = jobs.get(recovery_job_id)
    if not isinstance(record, dict):
        return
    if inference_state is not None:
        record["inference_state"] = inference_state
    if recovery_state is not None:
        record["recovery_state"] = recovery_state
    record["updated_at"] = datetime.now(UTC).isoformat()
    jobs[recovery_job_id] = record
    _write_seedvr2_recovery_payload(payload)


def _load_seedvr2_recovery_job(recovery_job_id: str) -> dict[str, Any]:
    payload = _read_seedvr2_recovery_payload()
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        raise HTTPException(status_code=404, detail="No SeedVR2 recovery metadata available.")
    record = jobs.get(recovery_job_id)
    if not isinstance(record, dict):
        raise HTTPException(status_code=404, detail=f"Unknown SeedVR2 recovery job: {recovery_job_id}")
    return record


def _list_seedvr2_recovery_jobs() -> list[dict[str, Any]]:
    payload = _read_seedvr2_recovery_payload()
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        return []
    rows = [value for value in jobs.values() if isinstance(value, dict)]
    rows.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return rows[:50]


def _run_manual_engine_preflight(
    *,
    transport: OpenSshTransport,
    connection: SshConnectionInfo,
    engine: EngineName,
) -> None:
    if engine == "seedvr2":
        command: list[str] = ["/opt/seedvr2-venv/bin/python", "/opt/prototype/seedvr2_prototype.py", "--help"]
    else:
        command = ["python", "-c", "import torch; from app.worker.process import run_pipeline; print(torch.__version__)"]
    try:
        result = transport.run_remote_command(
            connection,
            command=command,
            timeout_seconds=30.0,
            check=True,
        )
        if engine == "seedvr2" and "--input-is-excerpt" not in (result.stdout or ""):
            raise OrchestrationError(
                "SeedVR2 runner is incompatible with current web contract: '--input-is-excerpt' is missing. "
                "Publish and use a newer SeedVR2 image tag built from this branch."
            )
    except SshTransportError as exc:
        raise OrchestrationError(f"Engine preflight failed for {engine}: {exc}") from exc


def _check_manual_instance_image_compatibility(
    *,
    instance: InstanceSummary,
    requested_image: str | None,
    effective_worker_image: str | None,
    engine: EngineName,
    callbacks: RunCallbacks,
) -> dict[str, Any]:
    instance_id = getattr(instance, "instance_id", "n/a")
    gpu_name = getattr(instance, "gpu_name", None) or "unknown"
    ssh_host = getattr(instance, "ssh_host", None) or "unknown"
    actual_image = (getattr(instance, "image_name", None) or "").strip() or None
    expected_repo = _WORKER_IMAGE_REPO_BY_ENGINE[engine]
    expected_image = (requested_image or "").strip() or effective_worker_image
    metadata: dict[str, Any] = {
        "requested": (requested_image or "").strip() or None,
        "suggested_or_expected": expected_image,
        "instance_declared": actual_image,
        "expected_repository": expected_repo,
        "verification_level": "unverified",
    }
    if actual_image is None:
        callbacks.report_provisioning_status(
            {
                "phase": "image-compatibility",
                "instance_id": str(instance_id),
                "gpu": gpu_name,
                "host": ssh_host,
                "status": "unverified",
                "status_message": f"Instance image metadata unavailable; expected repository: {expected_repo}.",
            }
        )
        return metadata
    parsed_repo = _parse_image_repository(actual_image)
    if parsed_repo is None:
        callbacks.report_provisioning_status(
            {
                "phase": "image-compatibility",
                "instance_id": str(instance_id),
                "gpu": gpu_name,
                "host": ssh_host,
                "status": "unverified",
                "status_message": f"Could not parse instance image metadata '{actual_image}'.",
            }
        )
        return metadata
    if parsed_repo != expected_repo:
        raise OrchestrationError(
            f"Selected engine '{engine}' is incompatible with instance image '{actual_image}' (repo={parsed_repo}). "
            f"Expected repository: {expected_repo}."
        )
    metadata["verification_level"] = "repository-match"
    if expected_image and actual_image != expected_image:
        callbacks.report_provisioning_status(
            {
                "phase": "image-compatibility",
                "instance_id": str(instance_id),
                "gpu": gpu_name,
                "host": ssh_host,
                "status": "repo-match-tag-unverified",
                "status_message": (
                    f"Instance repository matches {expected_repo}, but exact tag differs "
                    f"(expected {expected_image}, actual {actual_image})."
                ),
            }
        )
        metadata["verification_level"] = "repository-match-tag-different"
    elif expected_image and actual_image == expected_image:
        metadata["verification_level"] = "exact-match"
    return metadata


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
        cleanup_excerpt_path: Path | None = None
        excerpt_result: VideoExcerptResult | None = None
        try:
            if request.source_name:
                original_video_path = _resolve_video_source_path(source_name=request.source_name)
            else:
                original_video_path = Path(request.input_path or "")
            video_path = original_video_path
            if request.excerpt_enabled:
                excerpt_dir = _excerpts_dir()
                excerpt_path = excerpt_dir / f"analysis-excerpt-{uuid4().hex[:16]}.mkv"
                excerpt_result = _create_excerpt_for_range(
                    source_path=original_video_path,
                    excerpt_path=excerpt_path,
                    start_timecode=request.excerpt_start,
                    requested_duration_seconds=request.excerpt_duration_seconds,
                )
                cleanup_excerpt_path = excerpt_result.excerpt_path
                video_path = excerpt_result.excerpt_path
            if not request.include_visual_diagnostics:
                analysis = analyze_source_video(video_path)
                payload = analysis.model_dump(mode="json")
            else:
                payload = analyze_source_video_report(
                video_path=video_path,
                include_visual_diagnostics=True,
                diagnostics_options=request.diagnostic_options,
                )
            payload["original_source"] = str(original_video_path)
            payload["excerpt_enabled"] = excerpt_result is not None
            payload["excerpt_start_seconds"] = excerpt_result.start_seconds if excerpt_result is not None else None
            payload["excerpt_requested_duration_seconds"] = (
                excerpt_result.requested_duration_seconds if excerpt_result is not None else None
            )
            payload["excerpt_actual_duration_seconds"] = excerpt_result.actual_duration_seconds if excerpt_result is not None else None
            payload["excerpt_source"] = str(excerpt_result.excerpt_path) if excerpt_result is not None else None
            return payload
        except SourceAnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            _cleanup_job_excerpt(cleanup_excerpt_path)

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
        payload = plan.model_dump(mode="json")
        payload["preview_dimensions"] = _preview_dimensions_from_plan_payload(payload)
        return payload

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

    @app.get("/api/seedvr2/recoveries")
    def api_seedvr2_recoveries() -> dict[str, Any]:
        return {"jobs": _list_seedvr2_recovery_jobs()}

    @app.post("/api/jobs/start")
    def api_start(request: StartJobRequest) -> dict[str, Any]:
        return manager.start_job(request)

    @app.post("/api/jobs/recover-seedvr2")
    def api_recover_seedvr2(request: Seedvr2RecoveryRequest) -> dict[str, Any]:
        recovery = _load_seedvr2_recovery_job(request.recovery_job_id)
        recovery_instance_id = int(recovery.get("instance_id"))
        if request.instance_id is not None and request.instance_id != recovery_instance_id:
            raise HTTPException(status_code=400, detail="Recovery job does not match the selected instance id.")
        profile = str(recovery.get("seedvr2_profile") or "smoke")
        input_path = str(recovery.get("local_input_path") or "").strip()
        output_path = str(recovery.get("local_output_path") or "").strip()
        report_path = str(recovery.get("local_report_path") or "").strip()
        start_request = StartJobRequest(
            mode="manual",
            engine="seedvr2",
            seedvr2_profile=profile,
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            instance_id=recovery_instance_id,
            recover_seedvr2_job_id=request.recovery_job_id,
        )
        return manager.start_job(start_request)

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
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Vast Upscaler — Interface locale</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --surface: #ffffff;
      --surface-alt: #f1f4fa;
      --text: #1a2233;
      --text-muted: #51607a;
      --border: #d7deea;
      --brand: #2f6feb;
      --brand-contrast: #ffffff;
      --success: #0a7a39;
      --warning: #9a6700;
      --danger: #b42318;
      --shadow: 0 6px 24px rgba(23, 35, 58, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.4;
    }

    .app {
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }

    .app-header {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }

    h1 {
      margin: 0 0 6px;
      font-size: 1.55rem;
      font-weight: 700;
    }

    h2 {
      margin: 0 0 10px;
      font-size: 1.18rem;
    }

    h3 {
      margin: 12px 0 8px;
      font-size: 1rem;
    }

    p { margin: 0; }

    .warn {
      margin-top: 8px;
      color: var(--warning);
      font-weight: 600;
    }

    form { margin: 0; }

    fieldset {
      margin: 0 0 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      background: var(--surface);
      box-shadow: var(--shadow);
      min-width: 0;
    }

    legend {
      font-weight: 700;
      padding: 0 8px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .step-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      font-size: 0.85rem;
      font-weight: 700;
    }

    .section-help {
      margin-top: 4px;
      color: var(--text-muted);
      font-size: 0.92rem;
    }

    label {
      display: block;
      margin-top: 10px;
      font-weight: 600;
      font-size: 0.93rem;
    }

    input, select {
      width: 100%;
      margin-top: 5px;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      background: #fff;
      color: var(--text);
      font-size: 0.95rem;
    }

    input:focus, select:focus, button:focus-visible, details summary:focus-visible {
      outline: 3px solid rgba(47, 111, 235, 0.24);
      outline-offset: 1px;
      border-color: var(--brand);
    }

    button {
      border: 1px solid var(--brand);
      background: var(--brand);
      color: var(--brand-contrast);
      border-radius: 10px;
      padding: 9px 12px;
      font-weight: 600;
      cursor: pointer;
      margin-top: 8px;
    }

    button.secondary {
      background: #fff;
      color: var(--brand);
    }

    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .row-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .muted { color: var(--text-muted); font-size: 0.9rem; }
    .status-line { margin-top: 8px; font-weight: 700; }
    .hidden { display: none !important; }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      font-size: 0.82rem;
      padding: 4px 9px;
      margin: 3px 6px 0 0;
      color: var(--text-muted);
    }

    .mode-note {
      margin-top: 8px;
      border: 1px dashed var(--border);
      background: var(--surface-alt);
      border-radius: 10px;
      padding: 10px;
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    .radio-group {
      margin-top: 8px;
      display: grid;
      gap: 8px;
    }

    .radio-option {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      margin: 0;
      cursor: pointer;
    }

    .radio-option input[type="radio"] {
      width: 18px;
      height: 18px;
      margin: 0;
      flex: 0 0 auto;
    }

    .recommendations-grid {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }

    .recommendation-card {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      background: var(--surface-alt);
    }

    .recommendation-head {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }

    .recommendation-title {
      margin: 0;
      font-size: 1rem;
      font-weight: 700;
    }

    .confidence {
      font-size: 0.82rem;
      border-radius: 999px;
      padding: 3px 8px;
      border: 1px solid var(--border);
      background: #fff;
    }

    .state-label {
      font-size: 0.82rem;
      font-weight: 700;
      border-radius: 999px;
      padding: 3px 8px;
      border: 1px solid transparent;
    }

    .state-recommande { color: #0f5132; background: #d1f4e0; border-color: #90e0b5; }
    .state-inutile { color: #2459a4; background: #dfeeff; border-color: #b8d2ff; }
    .state-incertain { color: #7a5600; background: #fff1cc; border-color: #f5d77d; }
    .state-indisponible { color: #7a7f8d; background: #eef0f4; border-color: #d8dde6; }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }

    .kpi {
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 10px;
      padding: 8px;
    }

    .kpi-label {
      color: var(--text-muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .kpi-value {
      margin-top: 4px;
      font-weight: 700;
      font-size: 0.95rem;
    }

    details {
      margin-top: 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
      padding: 8px 10px;
    }

    details summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--text);
      list-style: none;
    }

    details summary::-webkit-details-marker { display: none; }

    .result-box {
      margin-top: 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      padding: 10px;
    }

    .result-box strong { display: block; margin-bottom: 4px; }

    #events {
      background: #101726;
      color: #d9e2f3;
      padding: 10px;
      border-radius: 10px;
      white-space: pre-wrap;
      height: 260px;
      overflow: auto;
      margin-top: 8px;
      border: 1px solid #1f314f;
      font-size: 0.84rem;
      line-height: 1.45;
    }

    @media (max-width: 920px) {
      .row, .row-3, .kpi-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="app">
    <header class="app-header">
      <h1>Vast Upscaler (local)</h1>
      <p class="muted">Interface locale de traitement vidéo assisté Vast.ai (analyse et orchestration sur votre machine).</p>
      <p class="warn">Sécurité : cette interface reste locale (127.0.0.1) et n’expose pas vos clés API au navigateur.</p>
    </header>

    <form id="job-form">
    <fieldset>
      <legend><span class="step-badge">1</span>Choisir une vidéo</legend>
      <p class="section-help">Sélectionnez une vidéo du dossier <code>video/</code>. Aucun upload HTTP n’est effectué.</p>
      <label for="source_name">Vidéo source (dossier projet <code>video/</code>)
        <select id="source_name">
          <option value="">-- Sélectionner une vidéo --</option>
        </select>
      </label>
      <div class="actions">
        <button type="button" id="refresh-sources-btn" class="secondary">Actualiser</button>
      </div>
      <div id="video-sources-dir" class="muted"></div>
      <div id="selected-source-meta" class="muted"></div>
      <div id="video-selection-state" class="status-line muted"></div>
      <h3>Périmètre du traitement</h3>
      <div class="radio-group" role="radiogroup" aria-label="Périmètre du traitement">
        <label class="radio-option"><input type="radio" name="processing_scope" value="full_video" checked />Vidéo entière</label>
        <label class="radio-option"><input type="radio" name="processing_scope" value="test_excerpt" />Extrait de test</label>
      </div>
      <div id="excerpt-controls" class="hidden">
        <div class="row">
          <label for="excerpt_start">Début de l'extrait (MM:SS ou HH:MM:SS)
            <input id="excerpt_start" value="00:00" />
          </label>
          <label for="excerpt_duration_seconds">Durée (secondes)
            <input id="excerpt_duration_seconds" type="number" min="0.1" step="0.1" value="10" />
          </label>
        </div>
        <div id="excerpt-range" class="muted">Plage sélectionnée : n/a</div>
        <div id="excerpt-upload-note" class="warn">Mode extrait actif : seul l'extrait préparé localement sera transféré vers Vast.ai.</div>
      </div>
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">Résolution</div><div id="video-kpi-resolution" class="kpi-value">n/a</div></div>
        <div class="kpi"><div class="kpi-label">Durée</div><div id="video-kpi-duration" class="kpi-value">n/a</div></div>
        <div class="kpi"><div class="kpi-label">FPS</div><div id="video-kpi-fps" class="kpi-value">n/a</div></div>
        <div class="kpi"><div class="kpi-label">Taille</div><div id="video-kpi-size" class="kpi-value">n/a</div></div>
      </div>
      <details>
        <summary>Options avancées — chemin manuel</summary>
        <label for="input_path">Chemin vidéo manuel (dépannage / compatibilité)
          <input id="input_path" placeholder="C:\\\\path\\\\to\\\\video.mp4" />
        </label>
      </details>
    </fieldset>

    <fieldset>
      <legend><span class="step-badge">2</span>Analyser la source</legend>
      <p class="section-help">Analyse locale : FFprobe + diagnostics visuels (cropdetect, idet…). Aucune location Vast.ai à cette étape.</p>
      <div class="actions">
        <button type="button" id="analyze-btn">Analyser la vidéo sélectionnée</button>
      </div>
      <div id="analysis-status" class="status-line">En attente</div>
      <div id="analysis-summary" class="muted"></div>
    </fieldset>

    <fieldset>
      <legend><span class="step-badge">3</span>Examiner et valider les prétraitements</legend>
      <p class="section-help">Les recommandations sont issues du diagnostic. Seules les opérations compatibles worker sont exécutables à distance.</p>
      <h3>Prétraitements recommandés</h3>
      <div id="recommendation-status" class="muted">Lancez d’abord une analyse pour afficher les recommandations.</div>
      <div id="recommendations-table" class="recommendations-grid hidden">
        <div id="recommendations-body"></div>
      </div>

      <h3>Configuration du recadrage</h3>
      <div class="radio-group" role="radiogroup" aria-label="Mode de recadrage">
        <label class="radio-option"><input type="radio" name="crop_mode" value="preserve" checked />Conserver le cadrage original</label>
        <label class="radio-option"><input type="radio" name="crop_mode" value="auto_crop" />Appliquer le recadrage recommandé</label>
        <label class="radio-option"><input type="radio" name="crop_mode" value="custom_crop" />Définir des marges personnalisées</label>
      </div>
      <div class="row">
        <label for="crop_top">Marge haut (px) <input id="crop_top" type="number" min="0" value="0" /></label>
        <label for="crop_bottom">Marge bas (px) <input id="crop_bottom" type="number" min="0" value="0" /></label>
      </div>
      <div class="row">
        <label for="crop_left">Marge gauche (px) <input id="crop_left" type="number" min="0" value="0" /></label>
        <label for="crop_right">Marge droite (px) <input id="crop_right" type="number" min="0" value="0" /></label>
      </div>
      <div class="row-3">
        <div class="kpi"><div class="kpi-label">Dimensions originales</div><div id="crop-source-dimensions" class="kpi-value">n/a</div></div>
        <div class="kpi"><div class="kpi-label">Dimensions après crop</div><div id="crop-result-dimensions" class="kpi-value">n/a</div></div>
        <div class="kpi"><div class="kpi-label">Réduction pixels</div><div id="crop-pixel-reduction" class="kpi-value">n/a</div></div>
      </div>
      <div id="crop-dimensions" class="muted"></div>
      <div id="crop-applied-margins" class="muted"></div>
      <div id="preprocessing-plan-status" class="status-line">Aucun plan généré pour le moment.</div>
      <div class="muted">Exécutable à distance actuellement : recadrage. Les autres opérations restent informatives.</div>
    </fieldset>

    <fieldset>
      <legend><span class="step-badge">4</span>Configurer et lancer l’upscale</legend>
      <label for="engine">Moteur
        <select id="engine">
          <option value="realesrgan">Real-ESRGAN (stable)</option>
          <option value="seedvr2">SeedVR2 (expérimental, manuel)</option>
        </select>
      </label>
      <label for="seedvr2_profile">Profil SeedVR2
        <select id="seedvr2_profile">
          <option value="smoke">smoke</option>
          <option value="reference">reference</option>
        </select>
      </label>
      <label for="mode">Mode Vast.ai
        <select id="mode">
          <option value="automatic">Automatique (recherche, création, traitement, nettoyage)</option>
          <option value="manual">Manuel (instance existante, sans création/destruction auto)</option>
        </select>
      </label>
      <div id="mode-help" class="mode-note">
        Mode automatique : l’application orchestre le cycle complet selon la logique existante.
      </div>
      <div class="row">
        <label for="target_width">Largeur cible <input id="target_width" type="number" min="1" /></label>
        <label for="target_height">Hauteur cible <input id="target_height" type="number" min="1" /></label>
      </div>
      <label for="worker_image">Image Docker worker (tag/version)
        <input id="worker_image" />
      </label>
      <div class="muted">Le tag proposé est le dernier connu pour le moteur sélectionné. Le dépôt doit correspondre au moteur choisi. En mode manuel, changer ce champ ne modifie pas l’image d’une instance déjà démarrée.</div>
      <label for="gpu_model">Modèle GPU <select id="gpu_model"></select></label>
      <label for="sort_by">Tri des offres (mode automatique)
        <select id="sort_by">
          <option value="price">prix</option>
          <option value="reliability">fiabilité</option>
          <option value="performance">performance</option>
        </select>
      </label>
      <label for="instance_id">ID instance (obligatoire en mode manuel) <input id="instance_id" type="number" /></label>
      <details>
        <summary>Options avancées de lancement</summary>
        <label for="output_path">Chemin sortie (optionnel) <input id="output_path" /></label>
        <label for="report_path">Chemin rapport (optionnel) <input id="report_path" /></label>
        <label for="preprocessing_plan_path">Chemin plan de prétraitement (prioritaire sur le plan IHM)</label>
        <input id="preprocessing_plan_path" />
        <label for="offer_id">Offer ID (optionnel, mode automatique) <input id="offer_id" type="number" /></label>
        <label for="ssh_bootstrap_mode">Mode bootstrap SSH
          <select id="ssh_bootstrap_mode">
            <option value="tofu">tofu</option>
            <option value="strict">strict</option>
          </select>
        </label>
      </details>
    </fieldset>

    <div class="actions">
      <button type="submit">Lancer le traitement</button>
      <button type="button" id="cancel-btn" class="secondary">Annuler</button>
      <select id="recovery_job_id" class="secondary">
        <option value="">-- Récupération SeedVR2 (job distant) --</option>
      </select>
      <button type="button" id="recover-btn" class="secondary">Récupérer les résultats</button>
    </div>
    </form>

    <fieldset>
      <legend>Suivi du traitement</legend>
      <p class="section-help">Les états ci-dessous reflètent les événements backend (SSE) réellement disponibles.</p>
      <div class="row-3">
        <div class="kpi"><div class="kpi-label">État courant</div><div id="status" class="kpi-value">idle</div></div>
        <div class="kpi"><div class="kpi-label">Étape</div><div id="job-stage" class="kpi-value">n/a</div></div>
        <div class="kpi"><div class="kpi-label">Progression</div><div id="job-progress" class="kpi-value">n/a</div></div>
      </div>
      <div class="row">
        <div class="result-box">
          <strong>Durée écoulée</strong>
          <div id="job-elapsed" class="muted">n/a</div>
        </div>
        <div class="result-box">
          <strong>Résultat</strong>
          <div id="job-result" class="muted">Aucun résultat pour l’instant.</div>
        </div>
      </div>
      <div id="job-error" class="status-line hidden" style="color: var(--danger);"></div>
      <details>
        <summary>Journal technique (SSE)</summary>
        <div id="events"></div>
      </details>
    </fieldset>
  </main>

  <script>
    let currentJobId = null;
    let lastEventId = 0;
    let es = null;
    let promptDialogOpen = false;
    let latestAnalysisPayload = null;
    let latestPlanPayload = null;
    let analysisRequestToken = 0;
    let currentBusinessState = "idle";
    let latestWorkerImageByEngine = { realesrgan: "", seedvr2: "" };
    let latestRecoveryJobs = [];
    const expectedWorkerRepoByEngine = {
      realesrgan: "ghcr.io/davbg67/vast-video-upscaler-worker",
      seedvr2: "ghcr.io/davbg67/vast-video-upscaler-seedvr2",
    };
    let sourcesIndex = new Map();
    let currentJobStartedAt = null;
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

    function prettyDuration(seconds) {
      if (typeof seconds !== "number" || Number.isNaN(seconds) || seconds < 0) return "n/a";
      const rounded = Math.round(seconds);
      const h = Math.floor(rounded / 3600);
      const m = Math.floor((rounded % 3600) / 60);
      const s = rounded % 60;
      if (h > 0) return `${h}h ${m}m ${s}s`;
      if (m > 0) return `${m}m ${s}s`;
      return `${s}s`;
    }

    function prettyFps(fps) {
      if (typeof fps !== "number" || Number.isNaN(fps) || fps <= 0) return "n/a";
      return fps.toFixed(3).replace(/\\.000$/, "") + " fps";
    }

    function setStatusLabel(value) {
      document.getElementById("status").textContent = value || "n/a";
    }

    function renderRecoveryJobs() {
      const select = document.getElementById("recovery_job_id");
      select.innerHTML = '<option value="">-- Récupération SeedVR2 (job distant) --</option>';
      for (const item of latestRecoveryJobs) {
        const id = item.recovery_job_id || "";
        if (!id) continue;
        const state = `${item.inference_state || "unknown"} / ${item.recovery_state || "unknown"}`;
        const instance = item.instance_id != null ? `instance ${item.instance_id}` : "instance n/a";
        const label = `${id} (${instance}, ${state})`;
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = label;
        select.appendChild(opt);
      }
    }

    function setModeHelp() {
      const modeSelect = document.getElementById("mode");
      const mode = modeSelect.value;
      const engine = document.getElementById("engine").value;
      const box = document.getElementById("mode-help");
      if (engine === "seedvr2" && mode !== "manual") {
        modeSelect.value = "manual";
      }
      if (engine === "seedvr2") {
        box.textContent = "SeedVR2 est disponible en mode manuel uniquement. L’extrait est préparé localement puis exécuté sur l’instance sélectionnée.";
      } else if (modeSelect.value === "manual") {
        box.textContent = "Mode manuel : vous louez et gérez l’instance vous-même. L’application ne crée ni ne détruit automatiquement l’instance.";
      } else {
        box.textContent = "Mode automatique : l’application recherche une offre, crée l’instance, exécute le traitement puis applique la logique de nettoyage existante.";
      }
      document.getElementById("seedvr2_profile").disabled = engine !== "seedvr2";
      const automaticOption = document.querySelector("#mode option[value='automatic']");
      if (automaticOption) {
        automaticOption.disabled = engine === "seedvr2";
      }
    }

    function syncWorkerImageForEngine() {
      const engine = document.getElementById("engine").value;
      const workerImage = document.getElementById("worker_image");
      const expectedRepo = expectedWorkerRepoByEngine[engine] || "";
      const current = workerImage.value.trim();
      const suggested = (latestWorkerImageByEngine[engine] || "").trim();
      const extractRepo = (value) => {
        if (!value) return "";
        return value.split("@", 1)[0].split(":", 1)[0].trim().toLowerCase();
      };
      const currentRepo = extractRepo(current);
      if (!current) {
        workerImage.value = suggested;
        return;
      }
      if (currentRepo && expectedRepo && currentRepo !== expectedRepo) {
        workerImage.value = suggested;
      }
    }

    function readProcessingScope() {
      const selected = document.querySelector("input[name='processing_scope']:checked");
      return selected ? selected.value : "full_video";
    }

    function excerptEnabled() {
      return readProcessingScope() === "test_excerpt";
    }

    function parseExcerptStartToSeconds(value) {
      const text = String(value || "").trim();
      const hhmmss = /^(\\d{1,3}):(\\d{2}):(\\d{2})$/;
      const mmss = /^(\\d{1,3}):(\\d{2})$/;
      let match = text.match(hhmmss);
      if (match) {
        const h = Number(match[1]);
        const m = Number(match[2]);
        const s = Number(match[3]);
        if (m >= 60 || s >= 60) return null;
        return h * 3600 + m * 60 + s;
      }
      match = text.match(mmss);
      if (match) {
        const m = Number(match[1]);
        const s = Number(match[2]);
        if (s >= 60) return null;
        return m * 60 + s;
      }
      return null;
    }

    function formatHms(totalSeconds) {
      if (!Number.isFinite(Number(totalSeconds)) || Number(totalSeconds) < 0) return "n/a";
      const rounded = Math.floor(Number(totalSeconds));
      const h = Math.floor(rounded / 3600);
      const m = Math.floor((rounded % 3600) / 60);
      const s = rounded % 60;
      if (h > 0) {
        return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
      }
      return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    function excerptRequestPayload() {
      if (!excerptEnabled()) {
        return {
          excerpt_enabled: false,
          excerpt_start: null,
          excerpt_duration_seconds: 10,
        };
      }
      const startRaw = document.getElementById("excerpt_start").value.trim() || "00:00";
      const duration = Number(document.getElementById("excerpt_duration_seconds").value || 10);
      return {
        excerpt_enabled: true,
        excerpt_start: startRaw,
        excerpt_duration_seconds: duration,
      };
    }

    function updateExcerptControls() {
      const controls = document.getElementById("excerpt-controls");
      const rangeLabel = document.getElementById("excerpt-range");
      controls.classList.toggle("hidden", !excerptEnabled());
      if (!excerptEnabled()) {
        rangeLabel.textContent = "Plage sélectionnée : n/a";
        return;
      }
      const startRaw = document.getElementById("excerpt_start").value.trim() || "00:00";
      const startSeconds = parseExcerptStartToSeconds(startRaw);
      const duration = Number(document.getElementById("excerpt_duration_seconds").value || 10);
      if (startSeconds === null || !Number.isFinite(duration) || duration <= 0) {
        rangeLabel.textContent = "Plage sélectionnée : format invalide";
        return;
      }
      rangeLabel.textContent = `Plage sélectionnée : ${formatHms(startSeconds)} → ${formatHms(startSeconds + duration)}`;
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

    function setCropInputsDisabled(disabled) {
      for (const id of ["crop_top", "crop_bottom", "crop_left", "crop_right"]) {
        document.getElementById(id).disabled = disabled;
      }
    }

    function getEnabledCropRectFromPlan() {
      const operations = latestPlanPayload?.operations || [];
      const cropOperation = operations.find((op) => op.operation_id === "crop" && op.enabled === true);
      const rect = cropOperation?.parameters?.crop_rect;
      if (!rect) return null;
      if (!Number.isFinite(Number(rect.width)) || !Number.isFinite(Number(rect.height))) return null;
      return {
        x: Number(rect.x),
        y: Number(rect.y),
        width: Number(rect.width),
        height: Number(rect.height),
      };
    }

    function updateCropMetrics() {
      const sourceInfo = document.getElementById("crop-source-dimensions");
      const resultInfo = document.getElementById("crop-result-dimensions");
      const reductionInfo = document.getElementById("crop-pixel-reduction");
      const marginsInfo = document.getElementById("crop-applied-margins");
      if (!latestAnalysisPayload || !latestAnalysisPayload.width || !latestAnalysisPayload.height) {
        sourceInfo.textContent = "n/a";
        resultInfo.textContent = "n/a";
        reductionInfo.textContent = "n/a";
        marginsInfo.textContent = "Marges appliquées : n/a";
        return;
      }
      const srcW = Number(latestAnalysisPayload.width);
      const srcH = Number(latestAnalysisPayload.height);
      sourceInfo.textContent = `${srcW} x ${srcH}`;

      const cropRect = getEnabledCropRectFromPlan();
      let outW = srcW;
      let outH = srcH;
      let margins = { top: 0, bottom: 0, left: 0, right: 0 };
      if (cropRect) {
        outW = cropRect.width;
        outH = cropRect.height;
        margins = {
          top: cropRect.y,
          left: cropRect.x,
          right: Math.max(0, srcW - (cropRect.x + cropRect.width)),
          bottom: Math.max(0, srcH - (cropRect.y + cropRect.height)),
        };
      }
      resultInfo.textContent = `${outW} x ${outH}`;
      const srcPixels = srcW * srcH;
      const outPixels = outW * outH;
      const delta = srcPixels - outPixels;
      if (delta <= 0) {
        reductionInfo.textContent = "Aucune";
      } else {
        const pct = ((delta / srcPixels) * 100).toFixed(1);
        reductionInfo.textContent = `${delta.toLocaleString("fr-FR")} px (${pct}%)`;
      }
      marginsInfo.textContent = `Marges appliquées : haut ${margins.top}px, bas ${margins.bottom}px, gauche ${margins.left}px, droite ${margins.right}px`;
    }

    function updateCropDimensionsPreview() {
      const info = document.getElementById("crop-dimensions");
      if (!latestAnalysisPayload || !latestAnalysisPayload.width || !latestAnalysisPayload.height) {
        info.textContent = "Dimensions finales : n/a";
        updateCropMetrics();
        return;
      }
      const srcW = Number(latestAnalysisPayload.width);
      const srcH = Number(latestAnalysisPayload.height);
      const mode = readCropMode();
      const preview = latestPlanPayload?.preview_dimensions;
      if (preview && Number.isFinite(Number(preview.width)) && Number.isFinite(Number(preview.height))) {
        const planW = Number(preview.width);
        const planH = Number(preview.height);
        info.textContent = `Dimensions finales : ${planW} x ${planH} (issues du plan validé)`;
        updateCropMetrics();
        return;
      }
      if (mode === "auto_crop") {
        info.textContent = `Dimensions finales : ${srcW} x ${srcH} (en attente du plan validé ou recadrage non fiable)`;
        updateCropMetrics();
        return;
      }
      if (mode === "preserve") {
        info.textContent = `Dimensions finales : ${srcW} x ${srcH} (conservation du cadrage original)`;
        updateCropMetrics();
        return;
      }
      const m = cropMarginsFromInputs();
      const outW = srcW - m.left - m.right;
      const outH = srcH - m.top - m.bottom;
      info.textContent = `Dimensions finales : ${outW} x ${outH} (source : ${srcW} x ${srcH})`;
      updateCropMetrics();
    }

    function recommendationStateLabel(rec, op) {
      if (unsupportedRemoteOps.has(rec.key)) {
        return { text: "Indisponible (worker)", css: "state-indisponible" };
      }
      if (op?.enabled) {
        return { text: "Recommandé", css: "state-recommande" };
      }
      if (rec.recommended_action === "preserve" || rec.recommended_action === "avoid_deinterlace") {
        return { text: "Inutile", css: "state-inutile" };
      }
      if (rec.confidence === "low" || rec.confidence === "unknown") {
        return { text: "Incertain", css: "state-incertain" };
      }
      return { text: "Incertain", css: "state-incertain" };
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
          return `cropdetect : état=${crop.status || "unknown"}, marges H${margins.top || 0}/B${margins.bottom || 0}/G${margins.left || 0}/D${margins.right || 0}`;
        }
        return `cropdetect : état=${crop.status || "unknown"}`;
      }
      if (opKey === "deinterlace") {
        const idet = visual.idet || {};
        return `idet : classification=${idet.classification || "indeterminate"}, état=${idet.status || "unknown"}`;
      }
      if (opKey === "denoise" || opKey === "deblock") {
        const noise = visual.noise_artifacts || {};
        return `bruit/artéfacts : bruit=${noise.likely_noise || "unknown"}, compression=${noise.likely_compression_artifacts || "unknown"}, état=${noise.status || "unknown"}`;
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
        crop: "Recadrage des bandes noires",
        deinterlace: "Désentrelacement",
        denoise: "Réduction du bruit",
        deblock: "Réduction des artéfacts de compression",
      };

      for (const rec of plan.recommendations || []) {
        const op = byKey.get(rec.key);
        const state = recommendationStateLabel(rec, op);
        const block = document.createElement("article");
        block.className = "recommendation-card";
        const command = commandHintFor(rec.key, cropRect);
        const executable = unsupportedRemoteOps.has(rec.key) ? "Non exécutable à distance dans cette phase." : "Exécutable à distance.";
        const confidence = rec.confidence || "unknown";
        const confidenceLabel = confidence === "high" ? "élevée" : confidence === "medium" ? "moyenne" : confidence === "low" ? "faible" : "inconnue";
        block.innerHTML = `
          <div class="recommendation-head">
            <h4 class="recommendation-title">${labels[rec.key] || rec.key}</h4>
            <div>
              <span class="state-label ${state.css}">${state.text}</span>
              <span class="confidence">Confiance : ${confidenceLabel}</span>
            </div>
          </div>
          <div class="muted"><strong>Diagnostic</strong> : ${diagnosticSummaryFor(rec.key, analysis)}</div>
          <div class="muted"><strong>Recommandation</strong> : ${rec.recommended_action || "unknown"}</div>
          <div class="muted"><strong>Explication</strong> : ${rec.reason || "n/a"}</div>
          <div class="muted"><strong>Exécution worker</strong> : ${executable}</div>
          <details>
            <summary>Détails techniques (commande de validation)</summary>
            <code>${command}</code>
          </details>
        `;
        body.appendChild(block);
      }
      setRecommendationTableVisible((plan.recommendations || []).length > 0);
      const warnings = (plan.warnings || []).join(" | ");
      document.getElementById("recommendation-status").textContent = warnings
        ? `Recommandations prêtes. Avertissements : ${warnings}`
        : "Recommandations prêtes.";
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
        appendEvent("Échec de lecture des vidéos : " + (data.detail || JSON.stringify(data)));
        document.getElementById("video-selection-state").textContent = "Impossible de lire le dossier vidéo.";
        return;
      }
      const select = document.getElementById("source_name");
      select.innerHTML = '<option value="">-- Sélectionner une vidéo --</option>';
      sourcesIndex = new Map();
      for (const item of data.sources || []) {
        const opt = document.createElement("option");
        opt.value = item.name;
        const modified = item.modified_at ? new Date(item.modified_at).toLocaleString() : "n/a";
        opt.textContent = `${item.name} (${prettyBytes(item.size_bytes)}, ${modified})`;
        select.appendChild(opt);
        sourcesIndex.set(item.name, item);
      }
      document.getElementById("video-sources-dir").textContent = `Dossier source : ${data.directory}`;
      if ((data.sources || []).length === 0) {
        document.getElementById("video-selection-state").textContent = "Aucune vidéo détectée dans le dossier video/.";
      } else {
        document.getElementById("video-selection-state").textContent = `${data.sources.length} vidéo(s) disponible(s).`;
      }
      appendEvent(`Liste des vidéos actualisée : ${(data.sources || []).length} fichier(s) sélectionnable(s).`);
    }

    async function rebuildPlanFromCurrentAnalysis(token) {
      if (!latestAnalysisPayload) return;
      const planStatus = document.getElementById("preprocessing-plan-status");
      planStatus.textContent = "Génération du plan de prétraitement…";
      updateCropDimensionsPreview();
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
        planStatus.textContent = "Erreur de plan de prétraitement : " + (payload.detail || JSON.stringify(payload));
        setRecommendationTableVisible(false);
        updateCropDimensionsPreview();
        return;
      }
      latestPlanPayload = payload;
      updateCropDimensionsPreview();
      renderRecommendations(payload, latestAnalysisPayload);
      const enabled = (payload.operations || []).filter(op => op.enabled).map(op => op.operation_id);
      planStatus.textContent = enabled.length
        ? `Plan prêt. Opérations activées : ${enabled.join(", ")}`
        : "Plan prêt. Aucune opération activée (comportement inchangé).";
    }

    async function analyzeSelectedSource() {
      const sourceName = document.getElementById("source_name").value;
      const manualInput = document.getElementById("input_path").value.trim();
      if (!sourceName && !manualInput) {
        document.getElementById("analysis-status").textContent = "Aucune source sélectionnée.";
        return;
      }
      const token = ++analysisRequestToken;
      latestAnalysisPayload = null;
      latestPlanPayload = null;
      updateCropDimensionsPreview();
      setRecommendationTableVisible(false);
      document.getElementById("analysis-status").textContent = "Analyse en cours…";
      document.getElementById("analysis-summary").textContent = "";
      document.getElementById("recommendation-status").textContent = "Analyse en cours…";
      const response = await fetch("/api/video/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(Object.assign(
          {
            source_name: sourceName || null,
            input_path: sourceName ? null : manualInput,
            include_visual_diagnostics: true,
          },
          excerptRequestPayload(),
        )),
      });
      const payload = await response.json();
      if (token !== analysisRequestToken) return;
      if (!response.ok) {
        document.getElementById("analysis-status").textContent = "Échec de l’analyse.";
        document.getElementById("analysis-summary").textContent = payload.detail || JSON.stringify(payload);
        document.getElementById("video-kpi-resolution").textContent = "n/a";
        document.getElementById("video-kpi-duration").textContent = "n/a";
        document.getElementById("video-kpi-fps").textContent = "n/a";
        updateCropDimensionsPreview();
        return;
      }
      latestAnalysisPayload = payload;
      const dims = payload.width && payload.height ? `${payload.width} x ${payload.height}` : "inconnue";
      document.getElementById("analysis-status").textContent = "Analyse terminée.";
      document.getElementById("analysis-summary").textContent = `Codec : ${payload.video_codec || "inconnu"}, résolution : ${dims}, cadence : ${prettyFps(Number(payload.frame_rate_fps || 0))}`;
      if (payload.excerpt_enabled) {
        appendEvent(`Analyse de l'extrait local : ${payload.excerpt_start_seconds}s → ${Number(payload.excerpt_start_seconds || 0) + Number(payload.excerpt_actual_duration_seconds || 0)}s`);
      }
      document.getElementById("video-kpi-resolution").textContent = dims;
      document.getElementById("video-kpi-duration").textContent = prettyDuration(Number(payload.duration_seconds || 0));
      document.getElementById("video-kpi-fps").textContent = prettyFps(Number(payload.frame_rate_fps || 0));
      updateCropDimensionsPreview();
      await rebuildPlanFromCurrentAnalysis(token);
    }

    function selectedSourceChanged() {
      const selected = document.getElementById("source_name").value;
      const meta = sourcesIndex.get(selected);
      if (!selected || !meta) {
        document.getElementById("selected-source-meta").textContent = "";
        document.getElementById("video-selection-state").textContent = "Sélection invalide.";
        return;
      }
      const modified = meta.modified_at ? new Date(meta.modified_at).toLocaleString() : "n/a";
      document.getElementById("selected-source-meta").textContent = `Fichier sélectionné : ${meta.name} | ${prettyBytes(meta.size_bytes)} | modifié ${modified}`;
      document.getElementById("video-selection-state").textContent = excerptEnabled()
        ? "Source valide. Un extrait local sera préparé avant transfert Vast.ai."
        : "Source valide.";
      document.getElementById("video-kpi-size").textContent = prettyBytes(meta.size_bytes);
      analyzeSelectedSource();
    }

    async function loadDefaults() {
      const res = await fetch("/api/config");
      const cfg = await res.json();
      latestWorkerImageByEngine = cfg.worker_image_latest_by_engine || latestWorkerImageByEngine;
      latestRecoveryJobs = cfg.seedvr2_recovery_jobs || [];
      renderRecoveryJobs();
      document.getElementById("target_width").value = cfg.target_width;
      document.getElementById("target_height").value = cfg.target_height;
      const engineSelect = document.getElementById("engine");
      engineSelect.innerHTML = "";
      for (const engine of (cfg.engines || ["realesrgan", "seedvr2"])) {
        const opt = document.createElement("option");
        opt.value = engine;
        opt.textContent = engine === "seedvr2" ? "SeedVR2 (expérimental, manuel)" : "Real-ESRGAN (stable)";
        engineSelect.appendChild(opt);
      }
      engineSelect.value = cfg.default_engine || "realesrgan";
      const seedvr2Profile = document.getElementById("seedvr2_profile");
      seedvr2Profile.innerHTML = "";
      for (const profile of (cfg.seedvr2_profiles || ["smoke", "reference"])) {
        const opt = document.createElement("option");
        opt.value = profile;
        opt.textContent = profile;
        seedvr2Profile.appendChild(opt);
      }
      seedvr2Profile.value = "smoke";
      document.getElementById("worker_image").value = cfg.worker_image_latest_known;
      if (cfg.worker_image_required) {
        appendEvent("Aucun tag worker fiable par défaut n’est connu. Renseignez un tag explicite en mode automatique.");
      }
      const gpu = document.getElementById("gpu_model");
      gpu.innerHTML = "";
      for (const model of cfg.gpu_models) {
        const opt = document.createElement("option");
        opt.value = model;
        opt.textContent = model;
        gpu.appendChild(opt);
      }
      appendEvent("Tag worker par défaut Real-ESRGAN : " + (latestWorkerImageByEngine.realesrgan || "n/a"));
      appendEvent("Tag worker par défaut SeedVR2 : " + (latestWorkerImageByEngine.seedvr2 || "n/a"));
      syncWorkerImageForEngine();
      setModeHelp();
    }

    async function answerPrompt(jobId, prompt) {
      if (handledPromptIds.has(prompt.prompt_id)) return;
      handledPromptIds.add(prompt.prompt_id);
      promptDialogOpen = true;
      if (prompt.kind === "verified_fingerprint") {
        const fp = window.prompt(prompt.message + "\\n\\nEntrez l’empreinte SHA256 vérifiée indépendamment :");
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

    function updateJobSummaryFromEvent(data) {
      const type = data.type;
      const payload = data.payload || {};
      if (type === "status") {
        const state = payload.state || payload.message || "running";
        currentBusinessState = String(state || currentBusinessState);
        setStatusLabel(state);
        if (payload.message) {
          document.getElementById("job-stage").textContent = payload.message;
        }
        if (state === "inference_succeeded_recovering") {
          document.getElementById("job-result").textContent = "Inférence terminée. Récupération des artefacts en cours…";
        }
      } else if (type === "provisioning") {
        const phase = payload.phase || payload.status || "provisioning";
        document.getElementById("job-stage").textContent = String(phase);
      } else if (type === "instance") {
        document.getElementById("job-stage").textContent = String(payload.message || payload.state || "instance");
      } else if (type === "worker") {
        const stage = payload.stage || "worker";
        document.getElementById("job-stage").textContent = String(stage);
        const framesDone = payload.frames_done;
        const framesTotal = payload.frames_total;
        const fps = payload.fps_processing;
        if (Number.isFinite(Number(framesDone)) && Number.isFinite(Number(framesTotal)) && Number(framesTotal) > 0) {
          const pct = ((Number(framesDone) / Number(framesTotal)) * 100).toFixed(1);
          document.getElementById("job-progress").textContent = `${pct}% (${framesDone}/${framesTotal})`;
        } else if (Number.isFinite(Number(framesDone))) {
          document.getElementById("job-progress").textContent = `${framesDone} images traitées`;
        } else {
          document.getElementById("job-progress").textContent = "n/a";
        }
        if (Number.isFinite(Number(payload.elapsed_seconds))) {
          document.getElementById("job-elapsed").textContent = prettyDuration(Number(payload.elapsed_seconds));
        } else if (currentJobStartedAt) {
          const elapsed = (Date.now() - currentJobStartedAt) / 1000;
          document.getElementById("job-elapsed").textContent = prettyDuration(elapsed);
        }
        if (Number.isFinite(Number(fps))) {
          document.getElementById("analysis-summary").textContent = `Cadence de traitement en cours : ${prettyFps(Number(fps))}`;
        }
      } else if (type === "worker-log") {
        // Keep technical logs in the SSE stream; summary UI remains stage-focused.
      } else if (type === "result") {
        currentBusinessState = "completed";
        setStatusLabel("completed");
        document.getElementById("job-stage").textContent = String(payload.state || "completed");
        document.getElementById("job-progress").textContent = "Terminé";
        document.getElementById("job-elapsed").textContent = prettyDuration(Number(payload.duration_seconds || 0));
        const outputPath = payload.output_path || "n/a";
        const reportPath = payload.report_path || "n/a";
        document.getElementById("job-result").innerHTML = `Sortie : <code>${outputPath}</code><br/>Rapport : <code>${reportPath}</code>`;
      } else if (type === "error") {
        if (currentBusinessState === "inference_succeeded_recovering" || currentBusinessState === "failed_recovery") {
          setStatusLabel("failed_recovery");
          document.getElementById("job-result").textContent = "Inférence terminée mais récupération des artefacts échouée.";
        } else {
          setStatusLabel("failed");
          document.getElementById("job-result").textContent = "Traitement échoué.";
        }
        document.getElementById("job-error").classList.remove("hidden");
        document.getElementById("job-error").textContent = "Erreur : " + String(payload.message || "échec inconnu");
      } else if (type === "prompt") {
        document.getElementById("job-stage").textContent = "Confirmation requise";
      }
    }

    function watchEvents(jobId) {
      if (es) es.close();
      es = new EventSource(`/api/jobs/${jobId}/events?last_event_id=${lastEventId}`);
      es.onmessage = async (evt) => {
        const data = JSON.parse(evt.data);
        lastEventId = data.id;
        appendEvent(`[${data.type}] ${JSON.stringify(data.payload)}`);
        updateJobSummaryFromEvent(data);
        if (data.type === "prompt") {
          await maybeHandlePrompt(jobId, data.payload);
        }
        if (data.type === "result" || data.type === "error") {
          if (data.type === "result") {
            setStatusLabel("completed");
          } else if (currentBusinessState === "inference_succeeded_recovering" || currentBusinessState === "failed_recovery") {
            setStatusLabel("failed_recovery");
          } else {
            setStatusLabel("failed");
          }
        }
      };
    }

    document.getElementById("job-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const sourceName = document.getElementById("source_name").value || null;
      const manualInput = document.getElementById("input_path").value.trim();
      if (!sourceName && !manualInput) {
        appendEvent("Échec du démarrage : sélectionnez une source video/ ou fournissez un chemin manuel (options avancées).");
        return;
      }
      const manualPlanPath = document.getElementById("preprocessing_plan_path").value.trim() || null;
      const enabledRemoteCapable = (latestPlanPayload?.operations || []).some(
        (op) => op.enabled && !unsupportedRemoteOps.has(op.operation_id)
      );
      const payload = {
        mode: document.getElementById("mode").value,
        engine: document.getElementById("engine").value,
        seedvr2_profile: document.getElementById("seedvr2_profile").value,
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
        excerpt_enabled: excerptEnabled(),
        excerpt_start: excerptEnabled() ? (document.getElementById("excerpt_start").value.trim() || "00:00") : null,
        excerpt_duration_seconds: excerptEnabled() ? Number(document.getElementById("excerpt_duration_seconds").value || 10) : 10,
      };
      const response = await fetch("/api/jobs/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        appendEvent("Échec du démarrage : " + (data.detail || JSON.stringify(data)));
        document.getElementById("job-error").classList.remove("hidden");
        document.getElementById("job-error").textContent = "Erreur : " + (data.detail || "démarrage refusé");
        return;
      }
      if (manualPlanPath && latestPlanPayload) {
        appendEvent("Un plan manuel est défini : il priorise les choix de prétraitement de l’IHM.");
      }
      currentJobId = data.job_id;
      lastEventId = 0;
      currentJobStartedAt = Date.now();
      currentBusinessState = "running";
      handledPromptIds.clear();
      promptDialogOpen = false;
      document.getElementById("job-error").classList.add("hidden");
      document.getElementById("job-error").textContent = "";
      document.getElementById("job-result").textContent = "Traitement démarré…";
      document.getElementById("job-progress").textContent = "n/a";
      document.getElementById("job-stage").textContent = "Initialisation";
      setStatusLabel("running");
      appendEvent("Traitement démarré : " + currentJobId);
      watchEvents(currentJobId);
    });

    document.getElementById("recover-btn").addEventListener("click", async () => {
      const recoveryId = document.getElementById("recovery_job_id").value;
      if (!recoveryId) {
        appendEvent("Sélectionnez un job SeedVR2 à récupérer.");
        return;
      }
      const response = await fetch("/api/jobs/recover-seedvr2", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ recovery_job_id: recoveryId }),
      });
      const data = await response.json();
      if (!response.ok) {
        appendEvent("Échec de la récupération : " + (data.detail || JSON.stringify(data)));
        document.getElementById("job-error").classList.remove("hidden");
        document.getElementById("job-error").textContent = "Erreur : " + (data.detail || "récupération refusée");
        return;
      }
      currentJobId = data.job_id;
      lastEventId = 0;
      currentJobStartedAt = Date.now();
      currentBusinessState = "running";
      handledPromptIds.clear();
      promptDialogOpen = false;
      document.getElementById("job-error").classList.add("hidden");
      document.getElementById("job-error").textContent = "";
      document.getElementById("job-result").textContent = "Récupération des résultats démarrée…";
      document.getElementById("job-progress").textContent = "indéterminée";
      document.getElementById("job-stage").textContent = "Récupération";
      setStatusLabel("running");
      appendEvent("Récupération SeedVR2 démarrée : " + currentJobId + " (source=" + recoveryId + ")");
      watchEvents(currentJobId);
    });

    document.getElementById("cancel-btn").addEventListener("click", async () => {
      if (!currentJobId) return;
      await fetch(`/api/jobs/${currentJobId}/cancel`, {method: "POST"});
      appendEvent("Annulation demandée.");
    });

    document.getElementById("mode").addEventListener("change", setModeHelp);
    document.getElementById("engine").addEventListener("change", () => {
      syncWorkerImageForEngine();
      setModeHelp();
    });
    document.getElementById("refresh-sources-btn").addEventListener("click", loadVideoSources);
    document.getElementById("source_name").addEventListener("change", selectedSourceChanged);
    document.getElementById("analyze-btn").addEventListener("click", analyzeSelectedSource);
    for (const radio of document.querySelectorAll("input[name='processing_scope']")) {
      radio.addEventListener("change", async () => {
        updateExcerptControls();
        if (document.getElementById("source_name").value) {
          await analyzeSelectedSource();
        }
      });
    }
    document.getElementById("excerpt_start").addEventListener("input", updateExcerptControls);
    document.getElementById("excerpt_duration_seconds").addEventListener("input", updateExcerptControls);
    for (const radio of document.querySelectorAll("input[name='crop_mode']")) {
      radio.addEventListener("change", async () => {
        setCropInputsDisabled(readCropMode() !== "custom_crop");
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

    setCropInputsDisabled(true);
    setModeHelp();
    updateExcerptControls();
    loadDefaults();
    loadVideoSources();
  </script>
</body>
</html>
"""
