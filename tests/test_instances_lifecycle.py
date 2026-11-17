import pytest

from vast_upscaler.instances import (
    InstanceLifecycleError,
    InstanceOwnershipError,
    WaitOptions,
    create_instance_with_confirmation,
    destroy_owned_instance,
    reconcile_ambiguous_create,
    wait_until_instance_ready,
)
from vast_upscaler.session_store import SessionRecord, SessionStore
from vast_upscaler.vast_write_client import (
    CreateInstanceRequest,
    InstanceSummary,
    VastCreateAmbiguousError,
)


class StubWriteClient:
    def __init__(self):
        self.created = []
        self.destroyed = []
        self.summaries = []
        self.create_raises = None
        self.destroy_raises = None
        self.reconcile_id = None

    def create_instance(self, req):
        if self.create_raises:
            raise self.create_raises
        self.created.append(req)
        return 123456

    def get_instance(self, _instance_id):
        if not self.summaries:
            raise AssertionError("No summary queued")
        return self.summaries.pop(0)

    def destroy_instance(self, instance_id):
        if self.destroy_raises:
            raise self.destroy_raises
        self.destroyed.append(instance_id)
        return True

    def reconcile_ambiguous_create(self, *, label):
        _ = label
        return self.reconcile_id



def _running_summary(instance_id=123456):
    return InstanceSummary(
        instance_id=instance_id,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess-test",
        image_runtype="ssh_direct",
        ssh_host="ssh1.vast.ai",
        ssh_port=10123,
    )



def test_creation_requires_explicit_confirmation() -> None:
    client = StubWriteClient()
    session = SessionRecord(session_id="s1")

    with pytest.raises(InstanceLifecycleError):
        create_instance_with_confirmation(
            confirmed=False,
            client=client,
            session=session,
            req=CreateInstanceRequest(ask_id=42, image="ghcr.io/a/b:tag"),
        )

    assert client.created == []



def test_creation_success_persists_instance_id(tmp_path) -> None:
    client = StubWriteClient()
    session = SessionRecord(session_id="s2")
    store = SessionStore(path=tmp_path / "sessions.json")

    store.append(session)
    instance_id = create_instance_with_confirmation(
        confirmed=True,
        client=client,
        session=session,
        req=CreateInstanceRequest(ask_id=42, image="ghcr.io/a/b:tag", label="sess-2"),
        store=store,
    )

    saved = store.get("s2")
    assert instance_id == 123456
    assert saved is not None
    assert saved.instance_id == "123456"
    assert saved.instance_owned_by_session is True



def test_creation_ambiguous_timeout_sets_cleanup_required(tmp_path) -> None:
    client = StubWriteClient()
    client.create_raises = VastCreateAmbiguousError("ambiguous")
    session = SessionRecord(session_id="s3")
    store = SessionStore(path=tmp_path / "sessions.json")
    store.append(session)

    with pytest.raises(VastCreateAmbiguousError):
        create_instance_with_confirmation(
            confirmed=True,
            client=client,
            session=session,
            req=CreateInstanceRequest(ask_id=42, image="ghcr.io/a/b:tag", label="sess-3"),
            store=store,
        )

    saved = store.get("s3")
    assert saved is not None
    assert saved.state == "CLEANUP_REQUIRED"



def test_reconcile_after_ambiguous_create(tmp_path) -> None:
    client = StubWriteClient()
    client.reconcile_id = 777888
    session = SessionRecord(session_id="s4", state="CLEANUP_REQUIRED")
    store = SessionStore(path=tmp_path / "sessions.json")
    store.append(session)

    recovered = reconcile_ambiguous_create(client=client, session=session, label="sess-4", store=store)

    saved = store.get("s4")
    assert recovered == 777888
    assert saved is not None
    assert saved.instance_owned_by_session is True
    assert saved.state == "INSTANCE_CREATED"



def test_wait_ready_success() -> None:
    client = StubWriteClient()
    client.summaries = [_running_summary()]

    summary = wait_until_instance_ready(client=client, instance_id=123456, options=WaitOptions(timeout_seconds=0.1))
    assert summary.actual_status == "running"



def test_wait_ready_timeout(monkeypatch) -> None:
    client = StubWriteClient()
    client.summaries = [
        InstanceSummary(1, "loading", "running", "starting", "x", "ssh_direct", None, None),
        InstanceSummary(1, "loading", "running", "starting", "x", "ssh_direct", None, None),
        InstanceSummary(1, "loading", "running", "starting", "x", "ssh_direct", None, None),
    ]

    values = iter([0.0, 0.2, 0.4, 1.1])
    monkeypatch.setattr("vast_upscaler.instances.time.perf_counter", lambda: next(values))
    monkeypatch.setattr("vast_upscaler.instances.time.sleep", lambda _s: None)

    with pytest.raises(InstanceLifecycleError):
        wait_until_instance_ready(client=client, instance_id=1, options=WaitOptions(timeout_seconds=1.0, poll_interval_seconds=0.0))



def test_destroy_success(tmp_path) -> None:
    client = StubWriteClient()
    session = SessionRecord(session_id="s6")
    session.mark_instance_owned("123")
    session.transition_to("CLEANUP_PENDING")

    store = SessionStore(path=tmp_path / "sessions.json")
    store.append(session)

    ok = destroy_owned_instance(client=client, session=session, instance_id=123, store=store)
    saved = store.get("s6")

    assert ok is True
    assert saved is not None
    assert saved.state == "DESTROYED"



def test_destroy_failed_sets_cleanup_required(tmp_path) -> None:
    class StubDestroyFalse(StubWriteClient):
        def destroy_instance(self, instance_id):
            self.destroyed.append(instance_id)
            return False

    client = StubDestroyFalse()
    session = SessionRecord(session_id="s7")
    session.mark_instance_owned("123")
    session.transition_to("CLEANUP_PENDING")

    store = SessionStore(path=tmp_path / "sessions.json")
    store.append(session)

    ok = destroy_owned_instance(client=client, session=session, instance_id=123, store=store)
    saved = store.get("s7")

    assert ok is False
    assert saved is not None
    assert saved.state == "CLEANUP_REQUIRED"



def test_destroy_rejected_for_external_instance() -> None:
    client = StubWriteClient()
    session = SessionRecord(session_id="s8")
    session.mark_instance_owned("123")

    with pytest.raises(InstanceOwnershipError):
        destroy_owned_instance(client=client, session=session, instance_id=999)
