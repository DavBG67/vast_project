from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    session_id: str
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    state: str = "CREATED"
    input_path: str | None = None
    output_path: str | None = None
    offer_id: str | None = None
    instance_id: str | None = None
    hourly_price_usd: float | None = None
    error: str | None = None


class SessionStore:
    def __init__(self, path: Path | None = None) -> None:
        default_path = Path.home() / ".vast_upscaler" / "sessions.json"
        self.path = path or default_path

    def _read_all(self) -> list[SessionRecord]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [SessionRecord.model_validate(item) for item in raw]

    def _write_all(self, sessions: list[SessionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [session.model_dump(mode="json") for session in sessions]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
