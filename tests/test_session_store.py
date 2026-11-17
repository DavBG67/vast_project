import json
from pathlib import Path

import pytest

from vast_upscaler.session_store import (
    SessionRecord,
    SessionStateError,
    SessionStore,
    SessionStoreCorruptionError,
)


def test_legacy_session_record_is_still_compatible() -> None:
    legacy = {
        "session_id": "legacy-1",
        "started_at": "2026-11-01T10:00:00+00:00",
        "state": "CREATED",
        "input_path": "C:/in.mp4",
        "output_path": "C:/out.mp4",
        "instance_id": None,
    }

    record = SessionRecord.model_validate(legacy)

    assert record.session_id == "legacy-1"
    assert record.state == "CREATED"
    assert record.is_incomplete is True
    assert record.instance_owned_by_session is False


def test_valid_state_transition_sequence() -> None:
    record = SessionRecord(session_id="s-1")

    record.transition_to("OFFER_SELECTED")
    record.transition_to("CONFIRMED")
    record.transition_to("INSTANCE_REQUESTED")
    record.transition_to("INSTANCE_CREATED")
    record.transition_to("INSTANCE_READY")
    record.transition_to("UPLOADING")
    record.transition_to("PROCESSING")
    record.transition_to("DOWNLOADING")
    record.transition_to("VERIFYING")
    record.transition_to("COMPLETED")
    record.transition_to("CLEANUP_PENDING")
    record.transition_to("DESTROYED")

    assert record.is_terminal is True
    assert record.destroyed_at is not None


def test_invalid_transition_raises() -> None:
    record = SessionRecord(session_id="s-2")

    with pytest.raises(SessionStateError):
        record.transition_to("PROCESSING")


def test_auto_cleanup_rejects_external_instance() -> None:
    record = SessionRecord(session_id="s-3")
    record.mark_instance_owned("ins-owned")

    assert record.can_auto_cleanup("ins-other") is False
    assert record.can_auto_cleanup(None) is False
    assert record.can_auto_cleanup("ins-owned") is True



def test_store_atomic_write_and_readback(tmp_path: Path) -> None:
    store = SessionStore(path=tmp_path / "sessions.json")

    first = SessionRecord(session_id="s-4", state="CREATED")
    store.append(first)

    stored = store.get("s-4")
    assert stored is not None
    assert stored.session_id == "s-4"

    payload = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload[0]["session_id"] == "s-4"



def test_list_incomplete_excludes_destroyed_and_cancelled(tmp_path: Path) -> None:
    store = SessionStore(path=tmp_path / "sessions.json")

    active = SessionRecord(session_id="s-active", state="PROCESSING")
    destroyed = SessionRecord(session_id="s-destroyed", state="DESTROYED")
    cancelled = SessionRecord(session_id="s-cancelled", state="CANCELLED")

    store.append(active)
    store.append(destroyed)
    store.append(cancelled)

    incomplete = store.list_incomplete()
    assert [s.session_id for s in incomplete] == ["s-active"]



def test_cleanup_failure_state_requires_attention() -> None:
    record = SessionRecord(session_id="s-5")
    record.transition_to("FAILED")
    record.transition_to("CLEANUP_REQUIRED")

    assert record.needs_attention is True


def test_missing_store_file_returns_empty(tmp_path: Path) -> None:
    store = SessionStore(path=tmp_path / "sessions.json")
    assert store.list_recent() == []


def test_existing_valid_empty_list_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text("[]", encoding="utf-8")

    store = SessionStore(path=path)
    assert store.list_recent() == []


def test_corrupted_json_raises_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text("{not-json", encoding="utf-8")

    store = SessionStore(path=path)
    with pytest.raises(SessionStoreCorruptionError) as exc:
        store.list_recent()
    assert "verify active instances directly on Vast.ai" in str(exc.value)


def test_invalid_root_structure_raises_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text('{"sessions": []}', encoding="utf-8")

    store = SessionStore(path=path)
    with pytest.raises(SessionStoreCorruptionError):
        store.list_recent()


def test_invalid_record_shape_raises_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text('[{"state":"CREATED"}]', encoding="utf-8")

    store = SessionStore(path=path)
    with pytest.raises(SessionStoreCorruptionError):
        store.list_recent()
