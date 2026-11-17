"""Phase 4 instance lifecycle helpers (no direct CLI side effects)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from vast_upscaler.session_store import SessionRecord, SessionStore
from vast_upscaler.vast_write_client import (
    CreateInstanceRequest,
    InstanceSummary,
    VastCreateAmbiguousError,
    VastWriteClient,
)


class InstanceLifecycleError(RuntimeError):
    pass


class InstanceOwnershipError(InstanceLifecycleError):
    pass


@dataclass(frozen=True)
class WaitOptions:
    timeout_seconds: float = 300.0
    poll_interval_seconds: float = 5.0


def create_instance_with_confirmation(
    *,
    confirmed: bool,
    client: VastWriteClient,
    session: SessionRecord,
    req: CreateInstanceRequest,
    store: SessionStore | None = None,
) -> int:
    if not confirmed:
        raise InstanceLifecycleError("Creation blocked: explicit confirmation is required for paid operations.")

    session.transition_to("INSTANCE_REQUESTED", reason="Creation requested")
    try:
        instance_id = client.create_instance(req)
    except VastCreateAmbiguousError:
        session.transition_to("CLEANUP_REQUIRED", reason="Create timed out with ambiguous result")
        if store is not None:
            store.save(session)
        raise

    session.mark_instance_owned(instance_id, label=req.label)
    if store is not None:
        store.save(session)
    session.transition_to("INSTANCE_CREATED", reason="Instance created")
    if store is not None:
        store.save(session)
    return instance_id


def reconcile_ambiguous_create(
    *,
    client: VastWriteClient,
    session: SessionRecord,
    label: str,
    store: SessionStore | None = None,
) -> int | None:
    instance_id = client.reconcile_ambiguous_create(label=label)
    if instance_id is not None:
        session.mark_instance_owned(instance_id, label=label)
        if store is not None:
            store.save(session)
        if session.state == "CLEANUP_REQUIRED":
            session.transition_to("INSTANCE_CREATED", reason="Ambiguous create reconciled")
            if store is not None:
                store.save(session)
    return instance_id


def wait_until_instance_ready(*, client: VastWriteClient, instance_id: int, options: WaitOptions | None = None) -> InstanceSummary:
    opts = options or WaitOptions()
    deadline = time.perf_counter() + opts.timeout_seconds

    while True:
        summary = client.get_instance(instance_id)
        status = (summary.actual_status or "").lower()
        if status == "running":
            return summary
        if status in {"exited", "offline", "unknown"}:
            raise InstanceLifecycleError(
                "Instance entered a non-recoverable status before readiness: " + (summary.actual_status or "unknown")
            )

        if time.perf_counter() >= deadline:
            raise InstanceLifecycleError("Timed out waiting for instance readiness.")
        time.sleep(opts.poll_interval_seconds)


def destroy_owned_instance(
    *,
    client: VastWriteClient,
    session: SessionRecord,
    instance_id: int,
    store: SessionStore | None = None,
) -> bool:
    if not session.can_auto_cleanup(str(instance_id)):
        raise InstanceOwnershipError(
            "Refusing to destroy instance without verified session ownership."
        )

    try:
        ok = client.destroy_instance(instance_id)
    except Exception as exc:
        session.cleanup_error = str(exc)
        session.transition_to("CLEANUP_REQUIRED", reason="Destroy request raised exception")
        if store is not None:
            store.save(session)
        raise

    if ok:
        session.transition_to("DESTROYED", reason="Instance destroyed")
        if store is not None:
            store.save(session)
        return True

    session.cleanup_error = "Destroy call returned unsuccessful response"
    session.transition_to("CLEANUP_REQUIRED", reason="Destroy failed")
    if store is not None:
        store.save(session)
    return False
