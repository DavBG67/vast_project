from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from vast_upscaler.orchestrator import OrchestrationError
from vast_upscaler.webapp import (
    RunCallbacks,
    StartJobRequest,
    WorkflowExecutionResult,
    create_app,
)


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
secrets:
  vast_api_key: test-key
  ssh_private_key_path: C:/keys/id_ed25519
vast:
  gpu_models: [RTX 3090, RTX 4090]
video:
  target_width: 1280
  target_height: 720
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _wait_for_status(client: TestClient, job_id: str, terminal: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in terminal:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for statuses={terminal}")


def test_web_config_endpoint_and_root(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "Vast Upscaler (local)" in root.text

    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    body = cfg.json()
    assert body["target_width"] == 1280
    assert "vast_api_key" not in body
    assert body["worker_image_latest_known"]


def test_web_job_prompt_and_completion(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    output = tmp_path / "output.mp4"
    report = tmp_path / "report.json"

    def _auto_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        approved = callbacks.confirm_paid("Paid operation confirmation required")
        if not approved:
            raise OrchestrationError("Cancelled before paid operation.")
        callbacks.report_provisioning_status({"phase": "selected-offer", "status": "ok", "instance_id": "1", "gpu": "RTX 3090", "host": "n/a", "status_message": "ready"})
        callbacks.report_worker_progress({"kind": "worker", "stage": "upscale", "frames_done": 10, "frames_total": 20})
        return WorkflowExecutionResult(
            mode="automatic",
            session_id="sess-test",
            state="COMPLETED",
            output_path=str(output),
            report_path=str(report),
            duration_seconds=1.25,
            estimated_total_usd=0.42,
            instance_id=123,
            instance_state="DESTROYED",
            cleanup_verified=True,
            cleanup_warning=None,
        )

    app = create_app(config_path=config_path, auto_runner=_auto_runner)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "input_path": str(source),
            "output_path": str(output),
            "report_path": str(report),
            "worker_image": "ghcr.io/acme/worker:test",
        },
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    pending = _wait_for_status(client, job_id, {"running"})
    prompt = pending["pending_prompt"]
    assert prompt["kind"] == "paid_confirmation"

    answered = client.post(
        f"/api/jobs/{job_id}/prompts/{prompt['prompt_id']}",
        json={"approved": True},
    )
    assert answered.status_code == 200

    done = _wait_for_status(client, job_id, {"completed"})
    assert done["result"]["cleanup_verified"] is True
    assert done["result"]["estimated_total_usd"] == 0.42


def test_web_rejects_parallel_jobs(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    def _auto_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        for _ in range(30):
            callbacks.raise_if_cancelled()
            time.sleep(0.02)
        return WorkflowExecutionResult(
            mode="automatic",
            session_id="sess-test",
            state="COMPLETED",
            output_path=None,
            report_path=None,
            duration_seconds=0.5,
            estimated_total_usd=0.1,
            instance_id=None,
            instance_state="DESTROYED",
            cleanup_verified=True,
            cleanup_warning=None,
        )

    app = create_app(config_path=config_path, auto_runner=_auto_runner)
    client = TestClient(app)

    first = client.post("/api/jobs/start", json={"mode": "automatic", "input_path": str(source)})
    assert first.status_code == 200
    second = client.post("/api/jobs/start", json={"mode": "automatic", "input_path": str(source)})
    assert second.status_code == 409

    cancel = client.post(f"/api/jobs/{first.json()['job_id']}/cancel")
    assert cancel.status_code == 200
    finished = _wait_for_status(client, first.json()["job_id"], {"cancelled", "completed", "failed"})
    assert finished["status"] in {"cancelled", "completed"}
