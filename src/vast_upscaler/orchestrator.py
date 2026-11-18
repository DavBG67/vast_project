"""Phase 4E orchestration services assembling modules A-D."""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from vast_upscaler.config import Settings
from vast_upscaler.cost import CostEstimate, estimate_cost
from vast_upscaler.instances import (
    InstanceLifecycleError,
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
from vast_upscaler.provisioning_history import (
    ProvisioningAttemptTracker,
    ProvisioningHistoryCorruptionError,
    ProvisioningHistoryStore,
    ProvisioningOutcome,
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
from vast_upscaler.transfer import (
    OpenSshTransport,
    SshConnectionError,
    add_known_host_key,
    build_ssh_connection_info,
    fetch_ssh_ed25519_host_key,
    lookup_known_host_key,
    validate_local_ssh_prerequisites,
)
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


class SshEndpointProbeError(SshConnectionError):
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
    max_wait_ready_seconds: float = 600.0
    max_destroy_wait_seconds: float = 60.0
    poll_interval_seconds: float = 10.0
    processing_fps: float | None = None
    sort_by: str = "price"
    offer_id: int | None = None
    ssh_bootstrap_mode: Literal["tofu", "strict"] = "tofu"


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
    metrics_warning: str | None = None


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
    request_verified_fingerprint: Callable[[str], str | None] | None = None,
    report_provisioning_status: Callable[[dict[str, str]], None] | None = None,
    report_worker_progress: Callable[[dict[str, object]], None] | None = None,
) -> OrchestrationSummary:
    session = _create_session(options)
    session_store.append(session)
    tracker = ProvisioningAttemptTracker(session_id=session.session_id)
    history_store = ProvisioningHistoryStore()
    metrics_warning: str | None = None

    source_video = inspect_video(options.input_path)
    offer = _select_offer(read_client=read_client, settings=settings, options=options)
    tracker.set_offer(offer_id=offer.offer_id, gpu_name=offer.gpu_name, hourly_price_usd=offer.dph_total)
    tracker.record_transition("OFFER_SELECTED")
    if report_provisioning_status is not None:
        report_provisioning_status(
            {
                "phase": "selected-offer",
                "instance_id": "n/a",
                "gpu": offer.gpu_name,
                "host": "n/a",
                "status": f"offer_id={offer.offer_id}",
                "status_message": "Offer selected before paid confirmation",
            }
        )
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
            metrics_warning=metrics_warning,
        )

    validate_local_ssh_prerequisites(
        private_key_path=_resolve_private_key(settings=settings, options=options),
        known_hosts_path=options.known_hosts_path,
    )

    if not confirm_paid("Paid Vast.ai instance creation is required. Continue?"):
        session.transition_to("CANCELLED", reason="User cancelled before instance creation")
        session_store.save(session)
        raise OrchestrationError("Cancelled before paid operation.")

    session.transition_to("CONFIRMED", reason="User confirmed paid operation")
    tracker.record_transition("CONFIRMED")
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
    _validate_create_payload(request=request, write_client=write_client)
    tracker.mark_instance_requested()

    instance_id = create_instance_with_confirmation(
        confirmed=True,
        client=write_client,
        session=session,
        req=request,
        store=session_store,
    )
    tracker.mark_instance_created(instance_id)

    worker_result: RemoteWorkerResult | None = None
    primary_error: Exception | None = None
    outcome: ProvisioningOutcome = "WORKER_FAIL"
    provisioning_succeeded = False
    processing_succeeded = False
    cleanup_warning: str | None = None
    cleanup_verified = False

    try:
        first_status_reported = False

        def _on_instance_status(current: InstanceSummary) -> None:
            nonlocal first_status_reported
            tracker.observe_instance_status(current)
            if report_provisioning_status is None:
                return

            host = current.ssh_host or current.machine_id or "unknown"
            message = current.status_message or "n/a"
            report_provisioning_status(
                {
                    "instance_id": str(current.instance_id),
                    "gpu": current.gpu_name or offer.gpu_name,
                    "host": host,
                    "status": current.actual_status or "unknown",
                    "status_message": message,
                    "phase": "first" if not first_status_reported else "update",
                }
            )
            first_status_reported = True

        summary = wait_until_instance_ready(
            client=write_client,
            instance_id=instance_id,
            options=WaitOptions(
                timeout_seconds=options.max_wait_ready_seconds,
                poll_interval_seconds=options.poll_interval_seconds,
            ),
            on_status=_on_instance_status,
        )
        session.transition_to("INSTANCE_READY", reason="Instance ready for SSH")
        tracker.mark_instance_ready(summary)
        provisioning_succeeded = True
        outcome = "PROVISION_SUCCESS"
        session_store.save(session)
        _wait_for_ssh_tcp_availability(
            instance=summary,
            timeout_seconds=75.0,
            retry_delay_seconds=5.0,
            report_provisioning_status=report_provisioning_status,
            tracker=tracker,
        )

        connection = _build_trusted_connection(
            instance=summary,
            private_key_path=_resolve_private_key(settings=settings, options=options),
            known_hosts_path=options.known_hosts_path,
            confirm_ssh_trust=confirm_ssh_trust,
            bootstrap_mode=options.ssh_bootstrap_mode,
            request_verified_fingerprint=request_verified_fingerprint,
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
            progress_callback=report_worker_progress,
        )
        session.local_output_retrieved = True
        session.transition_to("DOWNLOADING", reason="Remote processing completed")
        session.transition_to("VERIFYING", reason="Local validation completed")
        session.transition_to("COMPLETED", reason="Workflow completed")
        processing_succeeded = True
        session_store.save(session)
    except RemoteWorkerAmbiguousStateError as exc:
        primary_error = exc
        outcome = "WORKER_FAIL"
        session.recovery_required = True
        session.recoverable_output_expected = True
        session.transition_to("PRESERVED_FOR_RECOVERY", reason=str(exc))
        session.error = str(exc)
        session_store.save(session)
    except KeyboardInterrupt as exc:
        primary_error = OrchestrationError("Local interruption received. Session preserved for manual recovery.")
        outcome = "LOCAL_INTERRUPT"
        session.recovery_required = True
        session.recoverable_output_expected = True
        session.transition_to("PRESERVED_FOR_RECOVERY", reason="Local interruption")
        session.error = str(exc)
        session_store.save(session)
    except Exception as exc:  # noqa: BLE001
        primary_error = exc
        if provisioning_succeeded:
            if isinstance(exc, SshTrustValidationError):
                outcome = "SSH_BOOTSTRAP_FAIL"
            elif isinstance(exc, SshConnectionError):
                outcome = "SSH_CONNECT_FAIL"
            else:
                outcome = "WORKER_FAIL"
        elif isinstance(exc, InstanceLifecycleError):
            if str(exc) == "Timed out waiting for instance readiness.":
                outcome = "PROVISION_TIMEOUT"
            else:
                outcome = "WORKER_FAIL"
        else:
            outcome = "WORKER_FAIL"
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
        metrics_warning = _persist_provisioning_history(
            history_store=history_store,
            tracker=tracker,
            outcome=outcome,
            provisioning_succeeded=provisioning_succeeded,
            processing_succeeded=processing_succeeded,
            report_provisioning_status=report_provisioning_status,
        )
        raise primary_error

    metrics_warning = _persist_provisioning_history(
        history_store=history_store,
        tracker=tracker,
        outcome=outcome,
        provisioning_succeeded=provisioning_succeeded,
        processing_succeeded=processing_succeeded,
        report_provisioning_status=report_provisioning_status,
    )

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
        metrics_warning=metrics_warning,
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
    request_verified_fingerprint: Callable[[str], str | None] | None = None,
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
            bootstrap_mode="tofu",
            request_verified_fingerprint=request_verified_fingerprint,
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
    normalized = normalize_offers(read_client.search_offers(query))
    compatible = filter_offers(normalized, constraints)
    offers = rank_offers(
        compatible,
        sort_by=options.sort_by,  # type: ignore[arg-type]
    )
    if not offers:
        raise OrchestrationError("No compatible offers found.")
    if options.offer_id is None:
        return offers[0]

    for offer in offers:
        if offer.offer_id == options.offer_id:
            return offer

    requested_seen = any(candidate.offer_id == options.offer_id for candidate in normalized)
    if requested_seen:
        raise OrchestrationError(
            f"Requested offer {options.offer_id} is incompatible with configured constraints. "
            "Refusing silent fallback to another offer."
        )
    raise OrchestrationError(
        f"Requested offer {options.offer_id} was not returned by Vast search. "
        "Refusing silent fallback to another offer."
    )


def _validate_create_payload(*, request: CreateInstanceRequest, write_client: VastWriteClient) -> None:
    mismatches = write_client.compare_create_payload_with_ssh_template(request)
    if mismatches:
        details = ", ".join(f"{key} ({value})" for key, value in sorted(mismatches.items()))
        raise OrchestrationError(
            "Create payload does not match expected SSH template parameters: "
            + details
            + "."
        )


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
    bootstrap_mode: Literal["tofu", "strict"],
    request_verified_fingerprint: Callable[[str], str | None] | None,
):
    if not instance.ssh_host or instance.ssh_port is None:
        raise SshTrustValidationError("Instance SSH host/port unavailable for trust validation.")

    known_key = lookup_known_host_key(
        known_hosts_path=known_hosts_path,
        host=instance.ssh_host,
        port=instance.ssh_port,
    )
    scanned = fetch_ssh_ed25519_host_key(host=instance.ssh_host, port=instance.ssh_port)

    if known_key is not None:
        if known_key.sha256_fingerprint != scanned.sha256_fingerprint:
            raise SshTrustValidationError(
                "SSH host key mismatch detected for approved host entry. "
                "Refusing connection and requiring manual investigation."
            )
    else:
        msg = (
            "SSH trust bootstrap required.\n"
            f"Host: {instance.ssh_host}:{instance.ssh_port}\n"
            f"Fingerprint (ED25519): {scanned.sha256_fingerprint}\n"
            "WARNING: This fingerprint is obtained from ssh-keyscan on first contact and is not independently "
            "verified by Vast.ai API/UI. Accepting it is Trust On First Use (TOFU)."
        )
        if not confirm_ssh_trust(msg):
            raise SshTrustValidationError("SSH trust bootstrap was not approved by operator.")

        if bootstrap_mode == "strict":
            if request_verified_fingerprint is None:
                raise SshTrustValidationError(
                    "Strict SSH bootstrap requires an independently verified fingerprint provider."
                )
            provided = (request_verified_fingerprint(msg) or "").strip()
            if not provided:
                raise SshTrustValidationError("No independently verified fingerprint provided by operator.")
            if provided != scanned.sha256_fingerprint:
                raise SshTrustValidationError(
                    "Provided independently verified fingerprint does not match scanned server fingerprint. "
                    "Refusing trust bootstrap."
                )
        add_known_host_key(known_hosts_path=known_hosts_path, host_key=scanned)

    return build_ssh_connection_info(
        instance=instance,
        private_key_path=private_key_path,
        known_hosts_path=known_hosts_path,
    )


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


def _persist_provisioning_history(
    *,
    history_store: ProvisioningHistoryStore,
    tracker: ProvisioningAttemptTracker,
    outcome: ProvisioningOutcome,
    provisioning_succeeded: bool,
    processing_succeeded: bool,
    report_provisioning_status: Callable[[dict[str, str]], None] | None,
) -> str | None:
    attempt = tracker.to_attempt(
        outcome=outcome,
        provisioning_succeeded=provisioning_succeeded,
        processing_succeeded=processing_succeeded,
    )
    try:
        history_store.append_attempt(attempt)
    except ProvisioningHistoryCorruptionError as exc:
        warning = (
            "Warning: provisioning metrics were not saved because the history file is corrupted. "
            f"{exc}"
        )
    except Exception as exc:  # noqa: BLE001
        warning = f"Warning: provisioning metrics were not saved ({exc})."
    else:
        return None

    if report_provisioning_status is not None:
        report_provisioning_status(
            {
                "phase": "metrics-warning",
                "instance_id": str(attempt.instance_id) if attempt.instance_id is not None else "n/a",
                "gpu": attempt.gpu_name or "unknown",
                "host": attempt.host_id or "unknown",
                "status": "warning",
                "status_message": warning,
            }
        )
    return warning


def _wait_for_ssh_tcp_availability(
    *,
    instance: InstanceSummary,
    timeout_seconds: float,
    retry_delay_seconds: float,
    report_provisioning_status: Callable[[dict[str, str]], None] | None,
    tracker: ProvisioningAttemptTracker,
) -> None:
    host = instance.ssh_host
    port = instance.ssh_port
    if not host:
        raise SshEndpointProbeError("Instance is running but SSH host is missing in Vast API response.")
    if port is None or port <= 0 or port > 65535:
        raise SshEndpointProbeError("Instance is running but SSH port is missing or invalid in Vast API response.")

    tracker.record_transition("SSH_ENDPOINT_WAIT_STARTED")
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            with socket.create_connection((host, port), timeout=5.0):
                tracker.record_transition("SSH_ENDPOINT_REACHABLE")
                if report_provisioning_status is not None:
                    report_provisioning_status(
                        {
                            "phase": "ssh-probe",
                            "instance_id": str(instance.instance_id),
                            "gpu": instance.gpu_name or "unknown",
                            "host": host,
                            "status": "reachable",
                            "status_message": f"SSH TCP endpoint reachable on attempt {attempt}",
                        }
                    )
                return
        except OSError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SshEndpointProbeError(
                    f"SSH TCP endpoint did not become reachable within {int(timeout_seconds)}s: {host}:{port}. "
                    f"Last error: {exc}"
                ) from exc

            if report_provisioning_status is not None:
                report_provisioning_status(
                    {
                        "phase": "ssh-probe",
                        "instance_id": str(instance.instance_id),
                        "gpu": instance.gpu_name or "unknown",
                        "host": host,
                        "status": "retrying",
                        "status_message": (
                            f"SSH TCP unavailable (attempt {attempt}): {exc}. "
                            f"Retry in {int(retry_delay_seconds)}s."
                        ),
                    }
                )
            time.sleep(retry_delay_seconds)


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
