from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vast_upscaler.vast_write_client import InstanceSummary

ProvisioningOutcome = Literal[
    "PROVISION_SUCCESS",
    "PROVISION_TIMEOUT",
    "SSH_BOOTSTRAP_FAIL",
    "SSH_CONNECT_FAIL",
    "WORKER_FAIL",
    "LOCAL_INTERRUPT",
]


class ProvisioningHistoryCorruptionError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class ProvisioningEvent(BaseModel):
    at: str
    kind: Literal["transition", "status"]
    name: str
    details: dict[str, str] = Field(default_factory=dict)


class ProvisioningAttempt(BaseModel):
    schema_version: int = 1
    session_id: str
    recorded_at: str = Field(default_factory=_utc_now_iso)
    started_at: str
    ended_at: str
    retention_expires_at: str
    outcome: ProvisioningOutcome
    provisioning_succeeded: bool
    processing_succeeded: bool
    provisioning_duration_seconds: float | None
    offer_id: int | None = None
    instance_id: int | None = None
    host_id: str | None = None
    machine_id: str | None = None
    gpu_name: str | None = None
    hourly_price_usd: float | None = None
    events: list[ProvisioningEvent] = Field(default_factory=list)


class ProvisioningHistoryDocument(BaseModel):
    schema_version: int = 1
    attempts: list[ProvisioningAttempt] = Field(default_factory=list)


class ProvisioningHistoryStore:
    def __init__(self, path: Path | None = None, retention_days: int = 90) -> None:
        self.path = path or (Path.home() / ".vast_upscaler" / "provisioning_history.json")
        self.retention_days = retention_days

    def append_attempt(self, attempt: ProvisioningAttempt) -> None:
        document = self._read_document()
        document.attempts = self._prune(document.attempts)
        document.attempts.append(attempt)
        self._write_document(document)

    def _read_document(self) -> ProvisioningHistoryDocument:
        if not self.path.exists():
            return ProvisioningHistoryDocument()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProvisioningHistoryCorruptionError(
                f"Provisioning history file is corrupted: {self.path}. "
                "Inspect or archive this file before retrying."
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Unable to read provisioning history file: {self.path}") from exc

        if not isinstance(raw, dict):
            raise ProvisioningHistoryCorruptionError(
                f"Provisioning history has invalid structure (expected object root): {self.path}."
            )

        try:
            return ProvisioningHistoryDocument.model_validate(raw)
        except Exception as exc:
            raise ProvisioningHistoryCorruptionError(
                f"Provisioning history contains invalid records: {self.path}."
            ) from exc

    def _write_document(self, document: ProvisioningHistoryDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(document.model_dump(mode="json"), indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(serialized)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self.path)

    def _prune(self, attempts: list[ProvisioningAttempt]) -> list[ProvisioningAttempt]:
        cutoff = _utc_now() - timedelta(days=self.retention_days)
        kept: list[ProvisioningAttempt] = []
        for attempt in attempts:
            ended_at = _parse_iso(attempt.ended_at)
            if ended_at >= cutoff:
                kept.append(attempt)
        return kept


@dataclass
class ProvisioningAttemptTracker:
    session_id: str
    started_at: datetime = field(default_factory=_utc_now)
    instance_requested_at: datetime | None = None
    instance_ready_at: datetime | None = None
    offer_id: int | None = None
    instance_id: int | None = None
    host_id: str | None = None
    machine_id: str | None = None
    gpu_name: str | None = None
    hourly_price_usd: float | None = None
    events: list[ProvisioningEvent] = field(default_factory=list)
    _last_status_signature: tuple[str, str, str, str] | None = None

    def record_transition(self, name: str, *, reason: str | None = None) -> None:
        details: dict[str, str] = {}
        if reason:
            details["reason"] = reason
        self.events.append(
            ProvisioningEvent(
                at=_utc_now_iso(),
                kind="transition",
                name=name,
                details=details,
            )
        )

    def set_offer(self, *, offer_id: int, gpu_name: str | None, hourly_price_usd: float | None) -> None:
        self.offer_id = offer_id
        self.gpu_name = gpu_name
        self.hourly_price_usd = hourly_price_usd

    def mark_instance_requested(self) -> None:
        self.instance_requested_at = _utc_now()
        self.record_transition("INSTANCE_REQUESTED")

    def mark_instance_created(self, instance_id: int) -> None:
        self.instance_id = instance_id
        self.record_transition("INSTANCE_CREATED")

    def observe_instance_status(self, summary: InstanceSummary) -> None:
        if summary.instance_id:
            self.instance_id = summary.instance_id
        if summary.machine_id:
            self.machine_id = summary.machine_id
            self.host_id = summary.machine_id
        if summary.gpu_name:
            self.gpu_name = summary.gpu_name

        signature = (
            summary.actual_status or "",
            summary.status_message or "",
            summary.ssh_host or "",
            str(summary.ssh_port or ""),
        )
        if signature == self._last_status_signature:
            return
        self._last_status_signature = signature

        details = {
            "status": summary.actual_status or "unknown",
            "status_message": summary.status_message or "n/a",
            "ssh_host": summary.ssh_host or "unknown",
            "ssh_port": str(summary.ssh_port) if summary.ssh_port is not None else "unknown",
        }
        self.events.append(
            ProvisioningEvent(
                at=_utc_now_iso(),
                kind="status",
                name="PROVISIONING_STATUS",
                details=details,
            )
        )

    def mark_instance_ready(self, summary: InstanceSummary) -> None:
        self.instance_ready_at = _utc_now()
        self.observe_instance_status(summary)
        self.record_transition("INSTANCE_READY")

    def to_attempt(
        self,
        *,
        outcome: ProvisioningOutcome,
        provisioning_succeeded: bool,
        processing_succeeded: bool,
    ) -> ProvisioningAttempt:
        ended_at = _utc_now()
        retention_expires = ended_at + timedelta(days=90)
        duration = self._compute_provisioning_duration(ended_at=ended_at)
        return ProvisioningAttempt(
            session_id=self.session_id,
            started_at=self.started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            retention_expires_at=retention_expires.isoformat(),
            outcome=outcome,
            provisioning_succeeded=provisioning_succeeded,
            processing_succeeded=processing_succeeded,
            provisioning_duration_seconds=duration,
            offer_id=self.offer_id,
            instance_id=self.instance_id,
            host_id=self.host_id,
            machine_id=self.machine_id,
            gpu_name=self.gpu_name,
            hourly_price_usd=self.hourly_price_usd,
            events=list(self.events),
        )

    def _compute_provisioning_duration(self, *, ended_at: datetime) -> float | None:
        if self.instance_requested_at is None:
            return None
        stop = self.instance_ready_at or ended_at
        seconds = (stop - self.instance_requested_at).total_seconds()
        return round(max(0.0, seconds), 3)
