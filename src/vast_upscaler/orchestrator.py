"""Phase 4E orchestration services assembling modules A-D."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from uuid import uuid4

from vast_upscaler.config import Settings
from vast_upscaler.cost import CostEstimate, estimate_cost
from vast_upscaler.instances import (
    WaitOptions,
    create_instance_with_confirmation,
    wait_until_instance_ready,
)
from vast_upscaler.offers import (
    OfferCandidate,
    OfferConstraints,
    build_offers_query,
    filter_offers,
    normalize_offers,
    rank_offers,
)
from vast_upscaler.remote_worker import (
    RemoteWorkerAmbiguousStateError,
    RemoteWorkerRequest,
    RemoteWorkerResult,
    execute_remote_worker,
    recover_remote_results,
)
from vast_upscaler.session_store import (
    SessionRecord,
    SessionStore,
    SessionStoreCorruptionError,
)
from vast_upscaler.transfer import OpenSshTransport, build_ssh_connection_info
from vast_upscaler.vast_client import VastClient
from vast_upscaler.vast_write_client import (
    CreateInstanceRequest,
    InstanceSummary,
    VastWriteClient,
)
from vast_upscaler.video import VideoInfo, inspect_video


class OrchestrationError(RuntimeError):
    pass


class SessionLockError(OrchestrationError):
    pass


class SshTrustValidationError(OrchestrationError):
    pass


@dataclass(frozen=True)
class OrchestrationOptions:
    input_path: Path
    output_path: Path
    report_path: Path
    worker_image: str
    profile: str
    target_width: int
    target_height: int
    dry_run: bool = False
    keep_instance: bool = False
    known_hosts_path: Path = Path.home() / ".ssh" / "known_hosts"
    ssh_private_key_path: Path | None = None
    max_wait_ready_seconds: float = 900.0
    max_destroy_wait_seconds: float = 60.0
    poll_interval_seconds: float = 5.0
    processing_fps: float | None = None
    sort_by: str = "price"


@dataclass(frozen=True)
class OrchestrationSummary:
    session_id: str
    state: str
    offer_id: int
    offer_gpu: str
    estimate: CostEstimate
    video_input: VideoInfo
    worker_result: RemoteWorkerResult | None
    instance_id: int | None
    cleanup_verified: bool
    cleanup_warning: str | None


@dataclass(frozen=True)
class ResumeSummary:
    session_id: str
    state: str
    message: str


@dataclass(frozen=True)
class CleanupItemResult:
    session_id: str
    attempted: bool
    cleaned: bool
    warning: str | None


class SessionExecutionLock:
    def __init__(self, lock_path: Path | None = None) -> None:
        self.path = lock_path or (Path.home() / ".vast_upscaler" / "workflow.lock")
        self._fd: int | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self._fd = os.open(str(self.path), flags)
        except FileExistsError as exc:
            raise SessionLockError(
                f"Another workflow appears to be running (lock file exists: {self.path}). "
                "Wait or clear stale lock after verification."
            ) from exc
        os.write(self._fd, str(os.getpid()).encode("ascii", errors="ignore"))
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            return


def run_upscale_workflow(
    *,
    settings: Settings,
    options: OrchestrationOptions,
    session_store: SessionStore,
    read_client: VastClient,
    write_client: VastWriteClient,
    transport: OpenSshTransport,
    confirm_paid: Callable[[str], bool],
    confirm_ssh_trust: Callable[[str], bool],
) -> OrchestrationSummary:
    session = _create_session(options)
    session_store.append(session)

    source_video = inspect_video(options.input_path)
    offer = _select_offer(read_client=read_client, settings=settings, options=options)
    session.offer_id = str(offer.offer_id)
    session.hourly_price_usd = offer.dph_total
    session.transition_to("OFFER_SELECTED", reason=f"Offer selected: {offer.offer_id}")
    session_store.save(session)

    estimate = estimate_cost(
        video=source_video,
        offer=offer,
        processing_fps_assumption=options.processing_fps,
        allocated_storage_gb=settings.vast.disk_gb,
    )
    _apply_cost_guardrails(session=session, settings=settings, estimate=estimate)
    session_store.save(session)

    if options.dry_run:
        return OrchestrationSummary(
            session_id=session.session_id,
            state=session.state,
            offer_id=offer.offer_id,
            offer_gpu=offer.gpu_name,
            estimate=estimate,
            video_input=source_video,
            worker_result=None,
            instance_id=None,
            cleanup_verified=False,
            cleanup_warning=None,
        )

    if not confirm_paid("Paid Vast.ai instance creation is required. Continue?"):
        session.transition_to("CANCELLED", reason="User cancelled before instance creation")
        session_store.save(session)
        raise OrchestrationError("Cancelled before paid operation.")

    session.transition_to("CONFIRMED", reason="User confirmed paid operation")
    session_store.save(session)

    request = CreateInstanceRequest(
        ask_id=offer.offer_id,
        image=options.worker_image,
        disk_gb=float(settings.vast.disk_gb),
        label=session.session_id,
        runtype="ssh_direct",
        target_state="running",
        cancel_unavail=True,
    )

    instance_id = create_instance_with_confirmation(
        confirmed=True,
        client=write_client,
        session=session,
        req=request,
        store=session_store,
    )

    worker_result: RemoteWorkerResult | None = None
    primary_error: Exception | None = None
    cleanup_warning: str | None = None
    cleanup_verified = False

    try:
        summary = wait_until_instance_ready(
            client=write_client,
            instance_id=instance_id,
            options=WaitOptions(
                timeout_seconds=options.max_wait_ready_seconds,
                poll_interval_seconds=options.poll_interval_seconds,
            ),
        )
        session.transition_to("INSTANCE_READY", reason="Instance ready for SSH")
        session_store.save(session)

        connection = _build_trusted_connection(
            instance=summary,
            private_key_path=_resolve_private_key(settings=settings, options=options),
            known_hosts_path=options.known_hosts_path,
            confirm_ssh_trust=confirm_ssh_trust,
        )
        transport.check_ssh_ready(connection, timeout_seconds=30.0)

        session.transition_to("PROCESSING", reason="Remote worker execution started")
        session_store.save(session)

        worker_result = execute_remote_worker(
            transport=transport,
            connection=connection,
            request=RemoteWorkerRequest(
                session_id=session.session_id,
                local_input_path=options.input_path,
                local_output_path=options.output_path,
                local_report_path=options.report_path,
                target_width=options.target_width,
                target_height=options.target_height,
                profile=options.profile,
            ),
        )
        session.local_output_retrieved = True
        session.transition_to("DOWNLOADING", reason="Remote processing completed")
        session.transition_to("VERIFYING", reason="Local validation completed")
        session.transition_to("COMPLETED", reason="Workflow completed")
        session_store.save(session)
    except RemoteWorkerAmbiguousStateError as exc:
        primary_error = exc
        session.recovery_required = True
        session.recoverable_output_expected = True
        session.transition_to("PRESERVED_FOR_RECOVERY", reason=str(exc))
        session.error = str(exc)
        session_store.save(session)
    except KeyboardInterrupt as exc:
        primary_error = OrchestrationError("Local interruption received. Session preserved for manual recovery.")
        session.recovery_required = True
        session.recoverable_output_expected = True
        session.transition_to("PRESERVED_FOR_RECOVERY", reason="Local interruption")
        session.error = str(exc)
        session_store.save(session)
    except Exception as exc:  # noqa: BLE001
        primary_error = exc
        session.transition_to("FAILED", reason=str(exc))
        session.error = str(exc)
        session_store.save(session)

    if not options.keep_instance and session.instance_id is not None:
        cleanup_verified, cleanup_warning = _cleanup_session_instance(
            session=session,
            store=session_store,
            client=write_client,
            max_wait_seconds=options.max_destroy_wait_seconds,
            poll_interval_seconds=options.poll_interval_seconds,
        )

    if primary_error is not None:
        if cleanup_warning:
            session.notes.append(f"Cleanup warning after primary failure: {cleanup_warning}")
            session_store.save(session)
        raise primary_error

    return OrchestrationSummary(
        session_id=session.session_id,
        state=session.state,
        offer_id=offer.offer_id,
        offer_gpu=offer.gpu_name,
        estimate=estimate,
        video_input=source_video,
        worker_result=worker_result,
        instance_id=instance_id,
        cleanup_verified=cleanup_verified,
        cleanup_warning=cleanup_warning,
    )


def list_active_instances(*, client: VastWriteClient) -> list[dict]:
    return client.list_instances(limit=25)


def destroy_instance_with_verification(
    *,
    client: VastWriteClient,
    instance_id: int,
    max_wait_seconds: float = 60.0,
    poll_interval_seconds: float = 5.0,
) -> bool:
    ok = client.destroy_instance(instance_id)
    if not ok:
        return False
    return _wait_until_instance_absent(
        client=client,
        instance_id=instance_id,
        timeout_seconds=max_wait_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def resume_session(
    *,
    store: SessionStore,
    session_id: str | None,
    write_client: VastWriteClient,
    settings: Settings,
    transport: OpenSshTransport,
    known_hosts_path: Path,
    ssh_private_key_path: Path | None,
    confirm_ssh_trust: Callable[[str], bool],
) -> ResumeSummary:
    session = store.get(session_id) if session_id else store.current()
    if session is None:
        raise OrchestrationError("No session found to resume.")
    if session.instance_id is None:
        raise OrchestrationError("Session has no managed instance id; cannot resume.")

    summary = write_client.get_instance(int(session.instance_id))
    status = (summary.actual_status or "unknown").lower()
    if status == "running":
        connection = _build_trusted_connection(
            instance=summary,
            private_key_path=ssh_private_key_path or _resolve_private_key(settings=settings, options=_resume_options()),
            known_hosts_path=known_hosts_path,
            confirm_ssh_trust=confirm_ssh_trust,
        )
        transport.check_ssh_ready(connection, timeout_seconds=30.0)

        if not session.output_path:
            raise OrchestrationError("Session has no local output path to recover to.")
        local_output = Path(session.output_path)
        local_report = local_output.with_suffix(local_output.suffix + ".report.json")
        local_input = Path(session.input_path) if session.input_path else local_output.with_name("source-placeholder.mkv")

        try:
            recover_remote_results(
                transport=transport,
                connection=connection,
                request=RemoteWorkerRequest(
                    session_id=session.session_id,
                    local_input_path=local_input,
                    local_output_path=local_output,
                    local_report_path=local_report,
                    target_width=settings.video.target_width,
                    target_height=settings.video.target_height,
                    profile=settings.video.default_profile,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            msg = "Recovery attempt could not confirm completed artifacts. Manual investigation required."
            if session.state != "PRESERVED_FOR_RECOVERY":
                session.transition_to("PRESERVED_FOR_RECOVERY", reason=msg)
            session.error = str(exc)
            store.save(session)
            return ResumeSummary(session_id=session.session_id, state=session.state, message=msg)

        session.local_output_retrieved = True
        if session.state != "PRESERVED_FOR_RECOVERY":
            session.transition_to("PRESERVED_FOR_RECOVERY", reason="Recovery artifacts downloaded")
        session.transition_to("DOWNLOADING", reason="Recovered remote artifacts")
        session.transition_to("VERIFYING", reason="Recovered artifacts validated")
        session.transition_to("COMPLETED", reason="Session resumed successfully")
        store.save(session)
        return ResumeSummary(
            session_id=session.session_id,
            state=session.state,
            message="Recovered completed artifacts without restarting processing.",
        )

    msg = f"Instance is not running (status={summary.actual_status or 'unknown'}). Manual reconciliation required."
    session.transition_to("PRESERVED_FOR_RECOVERY", reason=msg)
    store.save(session)
    return ResumeSummary(session_id=session.session_id, state=session.state, message=msg)


def cleanup_sessions(
    *,
    store: SessionStore,
    client: VastWriteClient,
    max_wait_seconds: float = 60.0,
    poll_interval_seconds: float = 5.0,
) -> list[CleanupItemResult]:
    results: list[CleanupItemResult] = []
    for session in store.list_incomplete():
        if session.state not in {"FAILED", "CLEANUP_PENDING", "CLEANUP_REQUIRED", "PRESERVED_FOR_RECOVERY"}:
            continue
        if not session.instance_id:
            results.append(CleanupItemResult(session_id=session.session_id, attempted=False, cleaned=False, warning=None))
            continue
        if not session.can_auto_cleanup(session.instance_id):
            results.append(
                CleanupItemResult(
                    session_id=session.session_id,
                    attempted=False,
                    cleaned=False,
                    warning="Refused cleanup for non-owned instance.",
                )
            )
            continue

        cleaned, warning = _cleanup_session_instance(
            session=session,
            store=store,
            client=client,
            max_wait_seconds=max_wait_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        results.append(
            CleanupItemResult(
                session_id=session.session_id,
                attempted=True,
                cleaned=cleaned,
                warning=warning,
            )
        )
    return results


def _select_offer(*, read_client: VastClient, settings: Settings, options: OrchestrationOptions) -> OfferCandidate:
    constraints = OfferConstraints(
        gpu_models=settings.vast.gpu_models,
        max_hourly_price_usd=settings.vast.max_hourly_price_usd,
        min_reliability=settings.vast.min_reliability,
        verified_only=settings.vast.verified_only,
        disk_gb=settings.vast.disk_gb,
    )
    query = build_offers_query(constraints=constraints, limit=100)
    offers = rank_offers(
        filter_offers(normalize_offers(read_client.search_offers(query)), constraints),
        sort_by=options.sort_by,  # type: ignore[arg-type]
    )
    if not offers:
        raise OrchestrationError("No compatible offers found.")
    return offers[0]


def _apply_cost_guardrails(*, session: SessionRecord, settings: Settings, estimate: CostEstimate) -> None:
    session.max_session_hours = settings.vast.max_session_hours
    session.estimated_total_cost_usd = estimate.total_estimated_usd
    if estimate.rental_total_hourly_usd is not None:
        session.known_cost_ceiling_usd = estimate.rental_total_hourly_usd * float(settings.vast.max_session_hours)
    if estimate.processing_hours_estimated is not None and estimate.processing_hours_estimated > settings.vast.max_session_hours:
        raise OrchestrationError(
            "Estimated processing duration exceeds configured max_session_hours. "
            "Local timeout is not a billing ceiling guarantee."
        )
    session.variable_cost_notes = list(estimate.notes)
    session.variable_cost_notes.append("Local timeout does not guarantee billing stop on provider side.")


def _resolve_private_key(*, settings: Settings, options: OrchestrationOptions) -> Path:
    if options.ssh_private_key_path is not None:
        return options.ssh_private_key_path
    if settings.secrets.ssh_private_key_path:
        return Path(settings.secrets.ssh_private_key_path)
    raise OrchestrationError("Missing SSH private key path. Set VAST_SSH_PRIVATE_KEY or provide explicit option.")


def _build_trusted_connection(
    *,
    instance: InstanceSummary,
    private_key_path: Path,
    known_hosts_path: Path,
    confirm_ssh_trust: Callable[[str], bool],
):
    if not known_hosts_path.exists() or not _known_host_contains(known_hosts_path, instance.ssh_host, instance.ssh_port):
        msg = (
            "SSH trust bootstrap required. Validate host fingerprint out-of-band, add it to known_hosts, "
            "then confirm to continue."
        )
        if not confirm_ssh_trust(msg):
            raise SshTrustValidationError("SSH trust not confirmed by operator.")

    return build_ssh_connection_info(
        instance=instance,
        private_key_path=private_key_path,
        known_hosts_path=known_hosts_path,
    )


def _known_host_contains(known_hosts_path: Path, host: str | None, port: int | None) -> bool:
    if host is None:
        return False
    text = known_hosts_path.read_text(encoding="utf-8")
    markers = [host]
    if port is not None:
        markers.append(f"[{host}]:{port}")
    return any(marker in text for marker in markers)


def _cleanup_session_instance(
    *,
    session: SessionRecord,
    store: SessionStore,
    client: VastWriteClient,
    max_wait_seconds: float,
    poll_interval_seconds: float,
) -> tuple[bool, str | None]:
    instance_id = int(session.instance_id or "0")
    if not session.can_auto_cleanup(session.instance_id):
        return False, "Refusing cleanup for non-owned instance."

    try:
        if session.state != "CLEANUP_PENDING":
            session.transition_to("CLEANUP_PENDING", reason="Auto cleanup requested")
            store.save(session)
        ok = client.destroy_instance(instance_id)
        if not ok:
            session.cleanup_error = "Destroy call returned unsuccessful response"
            session.transition_to("CLEANUP_REQUIRED", reason="Destroy API returned unsuccessful response")
            store.save(session)
            return False, session.cleanup_error
    except Exception as exc:  # noqa: BLE001
        return False, f"Destroy API call failed: {exc}"

    gone = _wait_until_instance_absent(
        client=client,
        instance_id=instance_id,
        timeout_seconds=max_wait_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    if not gone:
        session.cleanup_error = "Destroy requested but instance still appears active."
        session.transition_to("CLEANUP_REQUIRED", reason="Destroy could not be verified")
        store.save(session)
        return False, "Destroy requested but disappearance not verified."
    session.transition_to("DESTROYED", reason="Destroy verified by absence polling")
    store.save(session)
    return True, None


def _wait_until_instance_absent(
    *,
    client: VastWriteClient,
    instance_id: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> bool:
    deadline = time.perf_counter() + timeout_seconds
    while True:
        rows = client.list_instances(limit=25)
        if not any(_instance_matches(row, instance_id) for row in rows):
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(poll_interval_seconds)


def _instance_matches(row: dict, instance_id: int) -> bool:
    rid = row.get("id")
    try:
        return int(rid) == int(instance_id)
    except (TypeError, ValueError):
        return False


def _create_session(options: OrchestrationOptions) -> SessionRecord:
    sid = f"sess-{uuid4().hex[:12]}"
    return SessionRecord(
        session_id=sid,
        input_path=str(options.input_path),
        output_path=str(options.output_path),
    )


def _resume_options() -> OrchestrationOptions:
    return OrchestrationOptions(
        input_path=Path("resume-input-placeholder.mkv"),
        output_path=Path("resume-output-placeholder.mkv"),
        report_path=Path("resume-report-placeholder.json"),
        worker_image="resume-placeholder",
        profile="balanced",
        target_width=1920,
        target_height=1080,
    )


def ensure_store_readable(store: SessionStore) -> None:
    try:
        store.current()
    except SessionStoreCorruptionError as exc:
        raise OrchestrationError(
            "Session store is corrupted; refusing new paid workflow until manual reconciliation."
        ) from exc
