from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SessionState = Literal[
    "CREATED",
    "OFFER_SELECTED",
    "CONFIRMED",
    "INSTANCE_REQUESTED",
    "INSTANCE_CREATED",
    "INSTANCE_READY",
    "UPLOADING",
    "PROCESSING",
    "DOWNLOADING",
    "VERIFYING",
    "COMPLETED",
    "FAILED",
    "PRESERVED_FOR_RECOVERY",
    "CLEANUP_PENDING",
    "CLEANUP_REQUIRED",
    "DESTROYED",
    "CANCELLED",
]

TERMINAL_STATES: set[SessionState] = {"DESTROYED", "CANCELLED"}

ALLOWED_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    "CREATED": {"OFFER_SELECTED", "INSTANCE_REQUESTED", "CLEANUP_PENDING", "FAILED", "CANCELLED"},
    "OFFER_SELECTED": {"CONFIRMED", "FAILED", "CANCELLED"},
    "CONFIRMED": {"INSTANCE_REQUESTED", "FAILED", "CANCELLED"},
    "INSTANCE_REQUESTED": {"INSTANCE_CREATED", "FAILED", "CLEANUP_REQUIRED"},
    "INSTANCE_CREATED": {"INSTANCE_READY", "FAILED", "CLEANUP_PENDING", "CLEANUP_REQUIRED"},
    "INSTANCE_READY": {"UPLOADING", "PROCESSING", "FAILED", "CLEANUP_PENDING", "PRESERVED_FOR_RECOVERY"},
    "UPLOADING": {"PROCESSING", "FAILED", "CLEANUP_PENDING", "PRESERVED_FOR_RECOVERY"},
    "PROCESSING": {"DOWNLOADING", "FAILED", "CLEANUP_PENDING", "PRESERVED_FOR_RECOVERY"},
    "DOWNLOADING": {"VERIFYING", "FAILED", "CLEANUP_PENDING", "PRESERVED_FOR_RECOVERY"},
    "VERIFYING": {"COMPLETED", "FAILED", "CLEANUP_PENDING", "PRESERVED_FOR_RECOVERY"},
    "COMPLETED": {"CLEANUP_PENDING", "DESTROYED"},
    "FAILED": {"CLEANUP_PENDING", "CLEANUP_REQUIRED", "PRESERVED_FOR_RECOVERY"},
    "PRESERVED_FOR_RECOVERY": {"DOWNLOADING", "VERIFYING", "COMPLETED", "CLEANUP_PENDING", "CLEANUP_REQUIRED", "DESTROYED"},
    "CLEANUP_PENDING": {"DESTROYED", "CLEANUP_REQUIRED"},
    "CLEANUP_REQUIRED": {"INSTANCE_CREATED", "DESTROYED"},
    "DESTROYED": set(),
    "CANCELLED": set(),
}


class SessionStateError(RuntimeError):
    pass


class SessionStoreCorruptionError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SessionRecord(BaseModel):
    session_id: str
    started_at: str = Field(default_factory=_utc_now_iso)
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)
    state: SessionState = "CREATED"
    last_transition_at: str = Field(default_factory=_utc_now_iso)

    confirmed_at: str | None = None
    instance_requested_at: str | None = None
    instance_created_at: str | None = None
    instance_ready_at: str | None = None
    processing_started_at: str | None = None
    processing_completed_at: str | None = None
    cleanup_attempted_at: str | None = None
    destroyed_at: str | None = None

    input_path: str | None = None
    output_path: str | None = None
    offer_id: str | None = None
    instance_id: str | None = None
    hourly_price_usd: float | None = None

    max_session_hours: int | None = None
    estimated_total_cost_usd: float | None = None
    known_cost_ceiling_usd: float | None = None
    variable_cost_notes: list[str] = Field(default_factory=list)

    instance_owned_by_session: bool = False
    managed_instance_id: str | None = None
    managed_instance_label: str | None = None

    preserve_on_failure: bool = False
    recovery_required: bool = False
    recoverable_output_expected: bool = False
    local_output_retrieved: bool = False

    last_operation: str | None = None
    cleanup_error: str | None = None
    notes: list[str] = Field(default_factory=list)

    error: str | None = None

    def transition_to(self, new_state: SessionState, *, reason: str | None = None, at: str | None = None) -> None:
        if self.state == new_state:
            return

        allowed = ALLOWED_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise SessionStateError(f"Invalid transition: {self.state} -> {new_state}")

        ts = at or _utc_now_iso()
        old_state = self.state
        self.state = new_state
        self.updated_at = ts
        self.last_transition_at = ts

        if reason:
            self.notes.append(f"{ts} transition {old_state}->{new_state}: {reason}")

        if new_state == "CONFIRMED":
            self.confirmed_at = ts
        elif new_state == "INSTANCE_REQUESTED":
            self.instance_requested_at = ts
        elif new_state == "INSTANCE_CREATED":
            self.instance_created_at = ts
        elif new_state == "INSTANCE_READY":
            self.instance_ready_at = ts
        elif new_state == "PROCESSING" and self.processing_started_at is None:
            self.processing_started_at = ts
        elif new_state in {"DOWNLOADING", "VERIFYING", "COMPLETED"}:
            self.processing_completed_at = ts
        elif new_state in {"CLEANUP_PENDING", "CLEANUP_REQUIRED"}:
            self.cleanup_attempted_at = ts
        elif new_state == "DESTROYED":
            self.destroyed_at = ts

    def mark_instance_owned(self, instance_id: str, *, label: str | None = None) -> None:
        normalized_id = str(instance_id)
        self.instance_id = normalized_id
        self.managed_instance_id = normalized_id
        self.managed_instance_label = label
        self.instance_owned_by_session = True
        self.updated_at = _utc_now_iso()

    def can_auto_cleanup(self, instance_id: str | None) -> bool:
        if not self.instance_owned_by_session:
            return False
        if not instance_id:
            return False
        if not self.managed_instance_id:
            return False
        return instance_id == self.managed_instance_id

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_incomplete(self) -> bool:
        return not self.is_terminal

    @property
    def needs_attention(self) -> bool:
        return self.state in {"FAILED", "PRESERVED_FOR_RECOVERY", "CLEANUP_REQUIRED"}


class SessionStore:
    def __init__(self, path: Path | None = None) -> None:
        default_path = Path.home() / ".vast_upscaler" / "sessions.json"
        self.path = path or default_path

    def _read_all(self) -> list[SessionRecord]:
        if not self.path.exists():
            return []

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SessionStoreCorruptionError(
                f"Session store is corrupted: {self.path}. "
                "Do not continue blindly: verify active instances directly on Vast.ai before retrying."
            ) from exc

        if not isinstance(raw, list):
            raise SessionStoreCorruptionError(
                f"Session store has invalid structure (expected JSON list): {self.path}. "
                "Verify active instances directly on Vast.ai before retrying."
            )

        try:
            return [SessionRecord.model_validate(item) for item in raw]
        except Exception as exc:
            raise SessionStoreCorruptionError(
                f"Session store contains invalid session records: {self.path}. "
                "Verify active instances directly on Vast.ai before retrying."
            ) from exc

    def _write_all(self, sessions: list[SessionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [session.model_dump(mode="json") for session in sessions]
        serialized = json.dumps(payload, indent=2)

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

    def list_recent(self, limit: int = 10) -> list[SessionRecord]:
        records = self._read_all()
        return list(reversed(records[-limit:]))

    def current(self) -> SessionRecord | None:
        records = self._read_all()
        return records[-1] if records else None

    def append(self, record: SessionRecord) -> None:
        records = self._read_all()
        records.append(record)
        self._write_all(records)

    def save(self, record: SessionRecord) -> None:
        records = self._read_all()
        for index, existing in enumerate(records):
            if existing.session_id == record.session_id:
                record.updated_at = _utc_now_iso()
                records[index] = record
                self._write_all(records)
                return

        records.append(record)
        self._write_all(records)

    def get(self, session_id: str) -> SessionRecord | None:
        for record in self._read_all():
            if record.session_id == session_id:
                return record
        return None

    def list_incomplete(self) -> list[SessionRecord]:
        return [record for record in self._read_all() if record.is_incomplete]
