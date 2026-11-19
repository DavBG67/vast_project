from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from vast_upscaler.local_preprocessing import LocalExecutionReport
from vast_upscaler.orchestrator import OrchestrationError
from vast_upscaler.source_analysis import InterlaceInfo, SourceVideoAnalysis
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
    assert "worker_image_latest_known" in body
    assert body["worker_image_latest_known"] != "ghcr.io/acme/worker:test"


def test_web_video_analyze_endpoint(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"x")
    app = create_app(config_path=config_path)
    client = TestClient(app)

    monkeypatch.setattr(
        "vast_upscaler.webapp.analyze_source_video",
        lambda _path: SourceVideoAnalysis(
            file_path=str(video_path),
            width=1920,
            height=1080,
            frame_rate_fps=24.0,
            audio_tracks=[],
            subtitle_tracks=[],
            interlace=InterlaceInfo(
                field_order=None,
                interlaced=None,
                confidence="uncertain",
                note="ffprobe did not provide field_order.",
            ),
        ),
    )

    response = client.post("/api/video/analyze", json={"input_path": str(video_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 1920
    assert body["height"] == 1080
    assert body["interlace"]["confidence"] == "uncertain"


def test_web_video_analyze_endpoint_with_visual_diagnostics(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"x")
    app = create_app(config_path=config_path)
    client = TestClient(app)

    monkeypatch.setattr(
        "vast_upscaler.webapp.analyze_source_video_report",
        lambda **_kwargs: {
            "file_path": str(video_path),
            "analysis_report": {"version": "1.0", "visual_diagnostics": {"partial": False}},
        },
    )

    response = client.post(
        "/api/video/analyze",
        json={"input_path": str(video_path), "include_visual_diagnostics": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_report"]["version"] == "1.0"


def test_web_preprocessing_plan_from_analysis_payload_without_reanalysis(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)

    def _should_not_run(**_kwargs):
        raise AssertionError("analyze_source_video_report must not be called when analysis_payload is provided")

    monkeypatch.setattr("vast_upscaler.webapp.analyze_source_video_report", _should_not_run)

    response = client.post(
        "/api/video/preprocessing-plan",
        json={
            "analysis_payload": {
                "file_path": "sample.mp4",
                "width": 1280,
                "height": 720,
                "frame_rate_fps": 25.0,
                "duration_seconds": 10.0,
                "video_codec": "h264",
                "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
            },
            "user_choices": {"crop_mode": "preserve"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "1.0"
    assert payload["operations"][0]["operation_id"] == "crop"


def test_web_preprocessing_plan_accepts_user_custom_crop(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post(
        "/api/video/preprocessing-plan",
        json={
            "analysis_payload": {
                "file_path": "sample.mp4",
                "width": 1280,
                "height": 720,
                "frame_rate_fps": 25.0,
                "duration_seconds": 10.0,
                "video_codec": "h264",
                "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
            },
            "user_choices": {
                "crop_mode": "custom_crop",
                "custom_crop_margins": {"top": 10, "bottom": 10, "left": 20, "right": 20},
            },
        },
    )
    assert response.status_code == 200
    crop = next(item for item in response.json()["operations"] if item["operation_id"] == "crop")
    assert crop["enabled"] is True


def test_web_preprocessing_plan_rejects_invalid_custom_crop(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post(
        "/api/video/preprocessing-plan",
        json={
            "analysis_payload": {
                "file_path": "sample.mp4",
                "width": 640,
                "height": 360,
                "frame_rate_fps": 25.0,
                "duration_seconds": 10.0,
                "video_codec": "h264",
                "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
            },
            "user_choices": {
                "crop_mode": "custom_crop",
                "custom_crop_margins": {"top": 0, "bottom": 0, "left": 350, "right": 350},
            },
        },
    )
    assert response.status_code == 400


def test_web_preprocessing_execution_preview_endpoint(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)

    monkeypatch.setattr(
        "vast_upscaler.webapp.execute_preprocessing_plan",
        lambda **_kwargs: LocalExecutionReport(
            status="dry_run",
            source_path="input.mp4",
            destination_path="output.mp4",
            planned_operations=["crop"],
            applied_operations=["crop"],
            filter_chain="crop=100:100:0:0",
            ffmpeg_command=["ffmpeg", "..."],
            encoding={"video_codec": "libx264", "crf": 18},
            resolution_before={"width": 320, "height": 180},
            resolution_after={"width": 320, "height": 140},
            duration_seconds=0.0,
            warnings=[],
            ffprobe_validation=None,
        ),
    )

    response = client.post(
        "/api/video/preprocessing-execution/preview",
        json={
            "plan_payload": {
                "version": "1.0",
                "source_summary": {
                    "file_path": "sample.mp4",
                    "width": 320,
                    "height": 180,
                    "frame_rate_fps": 25.0,
                    "duration_seconds": 10.0,
                    "video_codec": "h264",
                    "interlace_metadata": {},
                    "diagnostics_available": False,
                },
                "recommendations": [],
                "user_choices": {
                    "crop_mode": "preserve",
                    "custom_crop_margins": None,
                    "deinterlace_mode": "preserve",
                    "denoise_mode": "preserve",
                    "deblock_mode": "preserve",
                },
                "operations": [
                    {
                        "operation_id": "crop",
                        "kind": "crop",
                        "enabled": True,
                        "recommended": False,
                        "confidence": "high",
                        "parameters": {
                            "mode": "auto_crop",
                            "crop_rect": {"x": 0, "y": 20, "width": 320, "height": 140},
                            "preserve_aspect_ratio": True,
                            "allow_stretch": False,
                            "fit_policy_after_crop": "letterbox_or_pad",
                        },
                        "justification": "manual",
                        "risk_notes": [],
                    }
                ],
                "warnings": [],
                "estimated_impact": {
                    "cpu_cost": "low",
                    "gpu_gain": "unknown",
                    "quality_risk": "low",
                    "note": "n/a",
                },
            },
            "output_path": str(tmp_path / "preview-out.mp4"),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "dry_run"


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
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-4ef981595f50",
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


def test_web_prompt_cannot_be_answered_twice_and_replay_is_safe(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    output = tmp_path / "output.mp4"
    report = tmp_path / "report.json"

    def _auto_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        approved = callbacks.confirm_paid("Paid operation confirmation required")
        if not approved:
            raise OrchestrationError("Cancelled before paid operation.")
        return WorkflowExecutionResult(
            mode="automatic",
            session_id="sess-test",
            state="COMPLETED",
            output_path=str(output),
            report_path=str(report),
            duration_seconds=1.0,
            estimated_total_usd=0.2,
            instance_id=10,
            instance_state="DESTROYED",
            cleanup_verified=True,
            cleanup_warning=None,
        )

    app = create_app(config_path=config_path, auto_runner=_auto_runner)
    client = TestClient(app)
    started = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "input_path": str(source),
            "output_path": str(output),
            "report_path": str(report),
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-4ef981595f50",
        },
    )
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    pending = _wait_for_status(client, job_id, {"running"})
    prompt_id = pending["pending_prompt"]["prompt_id"]

    first_answer = client.post(f"/api/jobs/{job_id}/prompts/{prompt_id}", json={"approved": True})
    assert first_answer.status_code == 200
    done = _wait_for_status(client, job_id, {"completed"})
    assert done["pending_prompt"] is None

    duplicate = client.post(f"/api/jobs/{job_id}/prompts/{prompt_id}", json={"approved": True})
    assert duplicate.status_code == 409

    replay = client.get(f"/api/jobs/{job_id}/events?last_event_id=0")
    assert replay.status_code == 200
    assert "prompt" in replay.text
    snapshot_after_replay = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot_after_replay["pending_prompt"] is None


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


def test_web_rejects_placeholder_worker_image_for_automatic_mode(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    app = create_app(config_path=config_path)
    client = TestClient(app)

    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "input_path": str(source),
            "worker_image": "ghcr.io/acme/worker:test",
        },
    )
    assert response.status_code == 400
    assert "placeholder worker image" in response.json()["detail"]


def test_manual_mode_keeps_instance_on_failure(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    def _manual_runner(_request: StartJobRequest, _settings, _callbacks: RunCallbacks) -> WorkflowExecutionResult:
        raise OrchestrationError("Worker report status is not completed: failed. Report notes: missing model.")

    app = create_app(config_path=config_path, manual_runner=_manual_runner)
    client = TestClient(app)
    started = client.post("/api/jobs/start", json={"mode": "manual", "input_path": str(source), "instance_id": 123})
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    terminal = _wait_for_status(client, job_id, {"failed"})
    assert terminal["status"] == "failed"
    assert "missing model" in terminal["error"]

    events_payload = client.get(f"/api/jobs/{job_id}/events?last_event_id=0").text
    assert "manual-instance-preserved" in events_payload


def test_manual_mode_keeps_instance_on_cancellation(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    def _manual_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        for _ in range(50):
            callbacks.raise_if_cancelled()
            time.sleep(0.02)
        return WorkflowExecutionResult(
            mode="manual",
            session_id=None,
            state="COMPLETED",
            output_path=None,
            report_path=None,
            duration_seconds=1.0,
            estimated_total_usd=None,
            instance_id=123,
            instance_state="running",
            cleanup_verified=False,
            cleanup_warning="Manual mode leaves instance running. Destroy manually when done.",
        )

    app = create_app(config_path=config_path, manual_runner=_manual_runner)
    client = TestClient(app)
    started = client.post("/api/jobs/start", json={"mode": "manual", "input_path": str(source), "instance_id": 123})
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    terminal = _wait_for_status(client, job_id, {"cancelled", "completed"})
    assert terminal["status"] in {"cancelled", "completed"}
    events_payload = client.get(f"/api/jobs/{job_id}/events?last_event_id=0").text
    assert "manual-instance-preserved" in events_payload or terminal["status"] == "completed"
