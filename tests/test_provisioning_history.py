from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vast_upscaler.provisioning_history import (
    ProvisioningAttempt,
    ProvisioningAttemptTracker,
    ProvisioningHistoryCorruptionError,
    ProvisioningHistoryStore,
)
from vast_upscaler.vast_write_client import InstanceSummary


def test_history_store_atomic_write_and_readback(tmp_path: Path) -> None:
    path = tmp_path / "provisioning_history.json"
    store = ProvisioningHistoryStore(path=path)
    tracker = ProvisioningAttemptTracker(session_id="sess-1")
    tracker.set_offer(offer_id=123, gpu_name="RTX 3090", hourly_price_usd=0.22)
    tracker.mark_instance_requested()
    tracker.mark_instance_created(555)
    tracker.mark_instance_ready(
        InstanceSummary(
            instance_id=555,
            actual_status="running",
            intended_status="running",
            cur_state="running",
            label="sess-1",
            image_runtype="ssh_direct",
            ssh_host="1.2.3.4",
            ssh_port=2200,
            gpu_name="RTX 3090",
            machine_id="host-1",
            status_message="ready",
        )
    )

    attempt = tracker.to_attempt(
        outcome="PROVISION_SUCCESS",
        provisioning_succeeded=True,
        processing_succeeded=True,
    )
    store.append_attempt(attempt)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert len(raw["attempts"]) == 1
    assert raw["attempts"][0]["session_id"] == "sess-1"
    assert raw["attempts"][0]["instance_id"] == 555


def test_history_store_retention_prunes_old_attempts(tmp_path: Path) -> None:
    path = tmp_path / "provisioning_history.json"
    now = datetime.now(UTC)
    stale_end = (now - timedelta(days=120)).isoformat()
    stale_start = (now - timedelta(days=121)).isoformat()
    fresh_end = (now - timedelta(days=10)).isoformat()
    fresh_start = (now - timedelta(days=11)).isoformat()
    document = {
        "schema_version": 1,
        "attempts": [
            ProvisioningAttempt(
                session_id="old",
                started_at=stale_start,
                ended_at=stale_end,
                retention_expires_at=(now - timedelta(days=30)).isoformat(),
                outcome="PROVISION_TIMEOUT",
                provisioning_succeeded=False,
                processing_succeeded=False,
                provisioning_duration_seconds=12.3,
            ).model_dump(mode="json"),
            ProvisioningAttempt(
                session_id="new",
                started_at=fresh_start,
                ended_at=fresh_end,
                retention_expires_at=(now + timedelta(days=80)).isoformat(),
                outcome="WORKER_FAIL",
                provisioning_succeeded=True,
                processing_succeeded=False,
                provisioning_duration_seconds=44.1,
            ).model_dump(mode="json"),
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    store = ProvisioningHistoryStore(path=path, retention_days=90)
    tracker = ProvisioningAttemptTracker(session_id="latest")
    attempt = tracker.to_attempt(
        outcome="LOCAL_INTERRUPT",
        provisioning_succeeded=False,
        processing_succeeded=False,
    )
    store.append_attempt(attempt)

    updated = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["session_id"] for item in updated["attempts"]]
    assert ids == ["new", "latest"]


def test_history_store_reports_corruption_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "provisioning_history.json"
    path.write_text("{not-json", encoding="utf-8")
    store = ProvisioningHistoryStore(path=path)
    tracker = ProvisioningAttemptTracker(session_id="sess-corrupt")
    attempt = tracker.to_attempt(
        outcome="LOCAL_INTERRUPT",
        provisioning_succeeded=False,
        processing_succeeded=False,
    )

    with pytest.raises(ProvisioningHistoryCorruptionError):
        store.append_attempt(attempt)


def test_tracker_skips_duplicate_poll_status() -> None:
    tracker = ProvisioningAttemptTracker(session_id="sess-dedupe")
    summary = InstanceSummary(
        instance_id=10,
        actual_status="loading",
        intended_status="running",
        cur_state="starting",
        label="sess-dedupe",
        image_runtype="ssh_direct",
        ssh_host="4.3.2.1",
        ssh_port=17022,
        gpu_name="RTX 3060",
        machine_id="host-42",
        status_message="pulling layer",
    )
    tracker.observe_instance_status(summary)
    tracker.observe_instance_status(summary)

    status_events = [event for event in tracker.events if event.kind == "status"]
    assert len(status_events) == 1
