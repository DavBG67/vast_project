from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import vast_upscaler.webapp as webapp_module
from vast_upscaler.local_preprocessing import LocalExecutionReport
from vast_upscaler.orchestrator import OrchestrationError
from vast_upscaler.remote_seedvr2_worker import (
    RemoteSeedvr2WorkerPreparationError,
    RemoteSeedvr2WorkerProcessError,
    RemoteSeedvr2WorkerRecoveryError,
)
from vast_upscaler.source_analysis import InterlaceInfo, SourceVideoAnalysis
from vast_upscaler.vast_write_client import InstanceSummary
from vast_upscaler.webapp import (
    RunCallbacks,
    StartJobRequest,
    WorkflowExecutionResult,
    _default_auto_runner,
    _default_manual_runner,
    _load_latest_known_worker_image,
    _preview_dimensions_from_plan_payload,
    _resolve_output_path,
    _resolve_remote_preprocessing_contract,
    _run_manual_engine_preflight,
    create_app,
)

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


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


def _read_events_text(client: TestClient, job_id: str) -> str:
    response = client.get(f"/api/jobs/{job_id}/events?last_event_id=0")
    assert response.status_code == 200
    return response.text


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
    assert body["default_engine"] == "realesrgan"
    assert set(body["engines"]) == {"realesrgan", "seedvr2"}
    assert set(body["seedvr2_profiles"]) == {"smoke", "reference"}


def test_web_index_contains_plan_based_dimension_preview_logic(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)
    root = client.get("/")
    assert root.status_code == 200
    html = root.text
    assert "latestPlanPayload?.preview_dimensions" in html
    assert "issues du plan validé" in html
    assert "en attente du plan validé ou recadrage non fiable" in html
    assert "currentRepo !== expectedRepo" in html
    assert "workerImage.value = suggested;" in html
    assert 'payload.state !== "manual-instance-preserved"' in html
    assert "MAX_EVENT_LINES = 1000" in html
    assert "resyncJobSnapshot" in html
    assert "scheduleSseReconnect" in html
    assert "localStorage.setItem" in html


def test_web_index_contains_required_ux_controls_and_sections(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)
    root = client.get("/")
    assert root.status_code == 200
    html = root.text
    assert "Choisir une vidéo" in html
    assert "Périmètre du traitement" in html
    assert "Extrait de test" in html
    assert "Analyser la source" in html
    assert "Examiner et valider les prétraitements" in html
    assert "Configurer et lancer l’upscale" in html
    for control_id in (
        "source_name",
        "excerpt_start",
        "excerpt_duration_seconds",
        "refresh-sources-btn",
        "analyze-btn",
        "recommendations-body",
        "crop_top",
        "crop_bottom",
        "crop_left",
        "crop_right",
        "engine",
        "seedvr2_profile",
        "mode",
        "worker_image",
        "status",
        "events",
        "cancel-btn",
    ):
        assert f'id="{control_id}"' in html


def test_web_video_sources_listing_filters_outputs_and_non_video(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "source.mp4").write_bytes(b"x")
    (video_dir / "clip.MKV").write_bytes(b"x")
    (video_dir / "source-upscaled.mp4").write_bytes(b"x")
    (video_dir / "source-cropped.mp4").write_bytes(b"x")
    (video_dir / "source-upscaled.mp4.report.json").write_text("{}", encoding="utf-8")
    (video_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    monkeypatch.setattr("vast_upscaler.webapp._project_root_path", lambda: tmp_path)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.get("/api/video/sources")
    assert response.status_code == 200
    payload = response.json()
    names = [item["name"] for item in payload["sources"]]
    assert "source.mp4" in names
    assert "clip.MKV" in names
    assert "source-upscaled.mp4" not in names
    assert "source-cropped.mp4" not in names
    assert all(not name.endswith(".report.json") for name in names)


def test_web_video_analyze_by_source_name(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    video_path = video_dir / "sample.mp4"
    video_path.write_bytes(b"x")

    monkeypatch.setattr("vast_upscaler.webapp._project_root_path", lambda: tmp_path)

    captured: dict[str, object] = {}

    def _fake_analyze(path: Path) -> SourceVideoAnalysis:
        captured["path"] = path
        return SourceVideoAnalysis(
            file_path=str(path),
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
        )

    monkeypatch.setattr("vast_upscaler.webapp.analyze_source_video", _fake_analyze)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post("/api/video/analyze", json={"source_name": "sample.mp4"})
    assert response.status_code == 200
    assert Path(str(captured["path"])).resolve() == video_path.resolve()


@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are required")
def test_web_video_analyze_uses_excerpt_when_enabled_and_cleans_temp_file(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    source = video_dir / "sample.mp4"
    build = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-t",
            "2.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        raise AssertionError(build.stderr)

    monkeypatch.setattr("vast_upscaler.webapp._project_root_path", lambda: tmp_path)
    captured: dict[str, Path] = {}

    def _fake_analyze_report(**kwargs):
        path = kwargs["video_path"]
        captured["path"] = path
        assert path.exists()
        return {
            "file_path": str(path),
            "width": 320,
            "height": 180,
            "duration_seconds": 1.0,
            "frame_rate_fps": 25.0,
            "video_codec": "h264",
            "analysis_report": {"visual_diagnostics": {}},
        }

    monkeypatch.setattr("vast_upscaler.webapp.analyze_source_video_report", _fake_analyze_report)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post(
        "/api/video/analyze",
        json={
            "source_name": "sample.mp4",
            "include_visual_diagnostics": True,
            "excerpt_enabled": True,
            "excerpt_start": "00:01",
            "excerpt_duration_seconds": 0.8,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["excerpt_enabled"] is True
    assert payload["original_source"].endswith("sample.mp4")
    excerpt_path = captured["path"]
    assert excerpt_path.parent == tmp_path / "data" / "excerpts"
    assert not excerpt_path.exists()


@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are required")
def test_web_job_start_rejects_excerpt_outside_source_duration(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    build = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25",
            "-t",
            "1.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        raise AssertionError(build.stderr)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "input_path": str(source),
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
            "excerpt_enabled": True,
            "excerpt_start": "00:00",
            "excerpt_duration_seconds": 3.0,
        },
    )
    assert response.status_code == 400
    assert "exceeds source duration" in response.json()["detail"]


def test_web_video_analyze_rejects_source_traversal(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "video").mkdir()
    monkeypatch.setattr("vast_upscaler.webapp._project_root_path", lambda: tmp_path)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post("/api/video/analyze", json={"source_name": "../secret.mp4"})
    assert response.status_code == 400
    assert "path separators" in response.json()["detail"]


def test_web_video_analyze_rejects_symlink_outside_video_directory(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    link = video_dir / "link.mp4"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        if os.name == "nt":
            pytest.skip("symlink creation is not permitted in this Windows environment")
        raise AssertionError("symlink creation should be available in CI Linux") from exc

    monkeypatch.setattr("vast_upscaler.webapp._project_root_path", lambda: tmp_path)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post("/api/video/analyze", json={"source_name": "link.mp4"})
    assert response.status_code == 400
    assert "outside the allowed directory" in response.json()["detail"]


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
    assert payload["preview_dimensions"] == {"width": 1280, "height": 720}


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
    assert response.json()["preview_dimensions"] == {"width": 1240, "height": 700}


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


def test_web_preprocessing_plan_auto_crop_uses_plan_effective_dimensions(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post(
        "/api/video/preprocessing-plan",
        json={
            "analysis_payload": {
                "file_path": "sample.mp4",
                "width": 640,
                "height": 480,
                "frame_rate_fps": 25.0,
                "duration_seconds": 10.0,
                "video_codec": "h264",
                "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
                "analysis_report": {
                    "version": "1.0",
                    "visual_diagnostics": {
                        "partial": False,
                        "cropdetect": {
                            "status": "ok",
                            "confidence": "high",
                            "recommended_crop": {"x": 6, "y": 36, "width": 628, "height": 408},
                            "probable_margins": {"top": 36, "bottom": 36, "left": 6, "right": 6},
                        },
                        "idet": {"status": "ok", "confidence": "high", "classification": "progressive"},
                        "noise_artifacts": {
                            "status": "ok",
                            "confidence": "low",
                            "likely_noise": "unknown",
                            "likely_compression_artifacts": "unknown",
                        },
                    },
                },
            },
            "user_choices": {"crop_mode": "auto_crop"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    crop = next(item for item in payload["operations"] if item["operation_id"] == "crop")
    assert crop["enabled"] is True
    assert crop["parameters"]["crop_rect"] == {"x": 6, "y": 36, "width": 628, "height": 408}
    assert payload["preview_dimensions"] == {"width": 628, "height": 408}


def test_preview_dimensions_helper_handles_preserve_auto_and_custom_cases() -> None:
    preserve_plan = {
        "source_summary": {"width": 640, "height": 480},
        "operations": [{"operation_id": "crop", "enabled": False, "parameters": {"crop_rect": None}}],
    }
    assert _preview_dimensions_from_plan_payload(preserve_plan) == {"width": 640, "height": 480}

    auto_plan = {
        "source_summary": {"width": 640, "height": 480},
        "operations": [
            {
                "operation_id": "crop",
                "enabled": True,
                "parameters": {"crop_rect": {"x": 6, "y": 36, "width": 628, "height": 408}},
            }
        ],
    }
    assert _preview_dimensions_from_plan_payload(auto_plan) == {"width": 628, "height": 408}

    custom_plan = {
        "source_summary": {"width": 640, "height": 480},
        "operations": [
            {
                "operation_id": "crop",
                "enabled": True,
                "parameters": {"crop_rect": {"x": 6, "y": 36, "width": 628, "height": 408}},
            }
        ],
    }
    assert _preview_dimensions_from_plan_payload(custom_plan) == {"width": 628, "height": 408}

    no_reliable_auto_plan = {
        "source_summary": {"width": 640, "height": 480},
        "operations": [{"operation_id": "crop", "enabled": False, "parameters": {"crop_rect": None}}],
    }
    assert _preview_dimensions_from_plan_payload(no_reliable_auto_plan) == {"width": 640, "height": 480}


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


def test_web_job_start_accepts_source_name_without_manual_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
    config_path = _write_config(tmp_path)
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    source = video_dir / "input.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr("vast_upscaler.webapp._project_root_path", lambda: tmp_path)

    def _auto_runner(_request: StartJobRequest, _settings, _callbacks: RunCallbacks) -> WorkflowExecutionResult:
        return WorkflowExecutionResult(
            mode="automatic",
            session_id="sess-test",
            state="COMPLETED",
            output_path=None,
            report_path=None,
            duration_seconds=0.1,
            estimated_total_usd=0.1,
            instance_id=None,
            instance_state="DESTROYED",
            cleanup_verified=True,
            cleanup_warning=None,
        )

    app = create_app(config_path=config_path, auto_runner=_auto_runner)
    client = TestClient(app)
    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "source_name": "input.mp4",
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
        },
    )
    assert response.status_code == 200


def test_resolve_output_path_defaults_to_mp4_for_all_sources(tmp_path: Path) -> None:
    mkv = tmp_path / "clip.mkv"
    webm = tmp_path / "clip.webm"
    mp4 = tmp_path / "clip.mp4"
    for item in (mkv, webm, mp4):
        item.write_bytes(b"x")
    for source in (mkv, webm, mp4):
        resolved = _resolve_output_path(request=StartJobRequest(mode="automatic", input_path=str(source))).name
        assert resolved.startswith("clip-upscaled-realesrgan-")
        assert resolved.endswith(".mp4")


def test_web_job_start_rejects_incompatible_explicit_output_extension(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mkv"
    source.write_bytes(b"x")
    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "input_path": str(source),
            "output_path": str(tmp_path / "output.webm"),
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
        },
    )
    assert response.status_code == 400
    assert "supports MP4/H.264/AAC output only" in response.json()["detail"]


def test_web_job_start_rejects_undecodable_source_before_runner(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.webm"
    source.write_bytes(b"x")

    called = {"runner": False}

    def _auto_runner(_request: StartJobRequest, _settings, _callbacks: RunCallbacks) -> WorkflowExecutionResult:
        called["runner"] = True
        return WorkflowExecutionResult(
            mode="automatic",
            session_id="sess",
            state="COMPLETED",
            output_path=None,
            report_path=None,
            duration_seconds=0.1,
            estimated_total_usd=0.1,
            instance_id=None,
            instance_state=None,
            cleanup_verified=True,
            cleanup_warning=None,
        )

    class _FakeRun:
        returncode = 1
        stdout = ""
        stderr = "Decoder not found for stream"

    monkeypatch.setattr("vast_upscaler.webapp.subprocess.run", lambda *args, **kwargs: _FakeRun())

    app = create_app(config_path=config_path, auto_runner=_auto_runner)
    client = TestClient(app)
    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "input_path": str(source),
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
        },
    )
    assert response.status_code == 400
    assert "decode precheck failed" in response.json()["detail"]
    assert called["runner"] is False


def test_web_job_start_rejects_source_name_plus_input_path(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    source = video_dir / "input.mp4"
    source.write_bytes(b"x")
    manual = tmp_path / "manual.mp4"
    manual.write_bytes(b"x")
    monkeypatch.setattr("vast_upscaler.webapp._project_root_path", lambda: tmp_path)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "source_name": "input.mp4",
            "input_path": str(manual),
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
        },
    )
    assert response.status_code == 400
    assert "either source_name or input_path" in response.json()["detail"]


def test_web_job_prompt_and_completion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
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


def test_web_prompt_cannot_be_answered_twice_and_replay_is_safe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
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


def test_web_rejects_parallel_jobs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
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
    request_payload = {
        "mode": "automatic",
        "input_path": str(source),
        "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-testparallel",
    }

    first = client.post("/api/jobs/start", json=request_payload)
    assert first.status_code == 200
    second = client.post("/api/jobs/start", json=request_payload)
    assert second.status_code == 409

    cancel = client.post(f"/api/jobs/{first.json()['job_id']}/cancel")
    assert cancel.status_code == 200
    finished = _wait_for_status(client, first.json()["job_id"], {"cancelled", "completed", "failed"})
    assert finished["status"] in {"cancelled", "completed"}


def test_web_rejects_placeholder_worker_image_for_automatic_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
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


def test_web_rejects_seedvr2_in_automatic_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    app = create_app(config_path=config_path)
    client = TestClient(app)

    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "automatic",
            "engine": "seedvr2",
            "input_path": str(source),
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-1234",
        },
    )
    assert response.status_code == 400
    assert "manual mode only" in response.json()["detail"]


def test_web_rejects_worker_image_repo_mismatch_for_selected_engine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    app = create_app(config_path=config_path)
    client = TestClient(app)

    response = client.post(
        "/api/jobs/start",
        json={
            "mode": "manual",
            "engine": "seedvr2",
            "input_path": str(source),
            "instance_id": 123,
            "worker_image": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-1",
        },
    )
    assert response.status_code == 400
    assert "repository mismatch" in response.json()["detail"]


def test_load_latest_worker_image_legacy_global_history_for_realesrgan(tmp_path: Path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps({"latest": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-legacy"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_module, "_WORKER_IMAGE_HISTORY_PATH", history_path)
    assert _load_latest_known_worker_image(engine="realesrgan") == "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-legacy"


def test_load_latest_worker_image_legacy_global_history_not_used_for_seedvr2(tmp_path: Path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps({"latest": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-legacy"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_module, "_WORKER_IMAGE_HISTORY_PATH", history_path)
    assert _load_latest_known_worker_image(engine="seedvr2") == ""


def test_load_latest_worker_image_per_engine_history_isolated(tmp_path: Path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "engines": {
                    "realesrgan": {"latest": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-r"},
                    "seedvr2": {"latest": "ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-x"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_module, "_WORKER_IMAGE_HISTORY_PATH", history_path)
    assert _load_latest_known_worker_image(engine="realesrgan").endswith(":dev-sha-r")
    assert _load_latest_known_worker_image(engine="seedvr2").endswith(":exp-6a2-x")


def test_load_latest_worker_image_rejects_incoherent_history_entry(tmp_path: Path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "engines": {
                    "seedvr2": {"latest": "ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-r"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_module, "_WORKER_IMAGE_HISTORY_PATH", history_path)
    assert _load_latest_known_worker_image(engine="seedvr2") == ""


def test_run_manual_engine_preflight_rejects_old_seedvr2_runner(monkeypatch) -> None:
    class _FakeTransport:
        def run_remote_command(self, _connection, *, command, **_kwargs):
            assert command == ["/opt/seedvr2-venv/bin/python", "/opt/prototype/seedvr2_prototype.py", "--help"]
            return SimpleNamespace(stdout="usage: seedvr2_prototype.py [-h] --input-video INPUT_VIDEO")

    with pytest.raises(OrchestrationError) as exc:
        _run_manual_engine_preflight(
            transport=_FakeTransport(),
            connection=object(),
            engine="seedvr2",
        )
    assert "--input-is-excerpt" in str(exc.value)


def test_manual_mode_keeps_instance_on_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
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


def test_manual_seedvr2_progress_does_not_revert_business_state_after_download_start(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    stage_emitted = threading.Event()
    continue_runner = threading.Event()

    def _manual_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        callbacks.report_worker_progress({"kind": "transfer", "stage": "download", "status": "started"})
        callbacks.report_worker_progress({"kind": "worker", "stage": "seedvr2-runner", "status": "started"})
        stage_emitted.set()
        continue_runner.wait(timeout=2.0)
        return WorkflowExecutionResult(
            mode="manual",
            session_id=None,
            state="COMPLETED",
            output_path=str(tmp_path / "out.mp4"),
            report_path=str(tmp_path / "report.json"),
            duration_seconds=0.1,
            estimated_total_usd=None,
            instance_id=123,
            instance_state="running",
            cleanup_verified=False,
            cleanup_warning="Manual mode leaves instance running. Destroy manually when done.",
        )

    app = create_app(config_path=config_path, manual_runner=_manual_runner)
    client = TestClient(app)
    started = client.post("/api/jobs/start", json={"mode": "manual", "engine": "seedvr2", "input_path": str(source), "instance_id": 123})
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    assert stage_emitted.wait(timeout=2.0)
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot["business_state"] == "inference_succeeded_recovering"
    continue_runner.set()
    terminal = _wait_for_status(client, job_id, {"completed"})
    assert terminal["business_state"] == "completed"


def test_manual_seedvr2_sequence_inference_success_then_download_then_completed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    def _manual_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        callbacks.report_worker_progress({"kind": "transfer", "stage": "download", "status": "started"})
        callbacks.report_worker_progress({"kind": "transfer", "stage": "download", "status": "completed"})
        return WorkflowExecutionResult(
            mode="manual",
            session_id=None,
            state="COMPLETED",
            output_path=str(tmp_path / "out.mp4"),
            report_path=str(tmp_path / "report.json"),
            duration_seconds=0.1,
            estimated_total_usd=None,
            instance_id=123,
            instance_state="running",
            cleanup_verified=False,
            cleanup_warning="Manual mode leaves instance running. Destroy manually when done.",
        )

    app = create_app(config_path=config_path, manual_runner=_manual_runner)
    client = TestClient(app)
    started = client.post("/api/jobs/start", json={"mode": "manual", "engine": "seedvr2", "input_path": str(source), "instance_id": 123})
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    terminal = _wait_for_status(client, job_id, {"completed"})
    assert terminal["status"] == "completed"
    events_payload = _read_events_text(client, job_id)
    recovering_index = events_payload.index('"state":"inference_succeeded_recovering"')
    download_started_index = events_payload.index('"stage":"download","status":"started"')
    download_completed_index = events_payload.index('"stage":"download","status":"completed"')
    result_index = events_payload.index('"type":"result"')
    assert recovering_index < download_started_index < download_completed_index < result_index


def test_manual_seedvr2_high_volume_worker_logs_preserve_terminal_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    def _manual_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        for index in range(10_000):
            callbacks.report_worker_progress({"kind": "worker-log", "line": f"log line {index}"})
        callbacks.report_worker_progress({"kind": "transfer", "stage": "download", "status": "started"})
        callbacks.report_worker_progress({"kind": "transfer", "stage": "download", "status": "completed"})
        return WorkflowExecutionResult(
            mode="manual",
            session_id=None,
            state="COMPLETED",
            output_path=str(tmp_path / "out.mp4"),
            report_path=str(tmp_path / "report.json"),
            duration_seconds=0.1,
            estimated_total_usd=None,
            instance_id=123,
            instance_state="running",
            cleanup_verified=False,
            cleanup_warning="Manual mode leaves instance running. Destroy manually when done.",
        )

    app = create_app(config_path=config_path, manual_runner=_manual_runner)
    client = TestClient(app)
    started = client.post("/api/jobs/start", json={"mode": "manual", "engine": "seedvr2", "input_path": str(source), "instance_id": 123})
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    terminal = _wait_for_status(client, job_id, {"completed"})
    assert terminal["status"] == "completed"
    assert terminal["latest_event_id"] >= terminal["oldest_event_id"] > 1
    assert (terminal["latest_event_id"] - terminal["oldest_event_id"]) < 5_000
    events_payload = _read_events_text(client, job_id)
    assert '"type":"result"' in events_payload


@pytest.mark.parametrize(
    ("exc", "expected_business_state"),
    [
        (RemoteSeedvr2WorkerPreparationError("prep failure"), "failed_preparation"),
        (RemoteSeedvr2WorkerProcessError("inference failure"), "failed_inference"),
        (RemoteSeedvr2WorkerRecoveryError("recovery failure"), "failed_recovery"),
    ],
)
def test_manual_seedvr2_exception_type_drives_terminal_business_state(
    tmp_path: Path,
    monkeypatch,
    exc: Exception,
    expected_business_state: str,
) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
    config_path = _write_config(tmp_path)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    def _manual_runner(_request: StartJobRequest, _settings, callbacks: RunCallbacks) -> WorkflowExecutionResult:
        callbacks.report_worker_progress({"kind": "transfer", "stage": "download", "status": "started"})
        raise exc

    app = create_app(config_path=config_path, manual_runner=_manual_runner)
    client = TestClient(app)
    started = client.post("/api/jobs/start", json={"mode": "manual", "engine": "seedvr2", "input_path": str(source), "instance_id": 123})
    assert started.status_code == 200
    terminal = _wait_for_status(client, started.json()["job_id"], {"failed"})
    assert terminal["business_state"] == expected_business_state


def test_manual_mode_keeps_instance_on_cancellation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vast_upscaler.webapp._validate_local_decode_capability", lambda **_kwargs: None)
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


def test_web_auto_and_manual_use_identical_preprocessing_contract(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    output = tmp_path / "output.mp4"
    report = tmp_path / "report.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        """
{
  "version": "1.0",
  "source_summary": {
    "file_path": "input.mp4",
    "width": 640,
    "height": 360,
    "frame_rate_fps": 24.0,
    "duration_seconds": 5.0,
    "video_codec": "h264",
    "interlace_metadata": {},
    "diagnostics_available": false
  },
  "recommendations": [],
  "user_choices": {
    "crop_mode": "custom_crop",
    "custom_crop_margins": {"top": 0, "bottom": 0, "left": 0, "right": 0},
    "deinterlace_mode": "preserve",
    "denoise_mode": "preserve",
    "deblock_mode": "preserve"
  },
  "operations": [
    {
      "operation_id": "crop",
      "kind": "crop",
      "enabled": true,
      "recommended": false,
      "confidence": "high",
      "parameters": {
        "mode": "custom_crop",
        "crop_rect": {"x": 0, "y": 0, "width": 640, "height": 360},
        "preserve_aspect_ratio": true,
        "allow_stretch": false,
        "fit_policy_after_crop": "letterbox_or_pad"
      },
      "justification": "manual",
      "risk_notes": []
    }
  ],
  "warnings": [],
  "estimated_impact": {
    "cpu_cost": "low",
    "gpu_gain": "unknown",
    "quality_risk": "low",
    "note": "n/a"
  }
}
""".strip(),
        encoding="utf-8",
    )

    request = StartJobRequest(
        mode="automatic",
        input_path=str(source),
        output_path=str(output),
        report_path=str(report),
        worker_image="ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
        preprocessing_plan_path=str(plan),
        instance_id=123,
    )
    settings = _write_config(tmp_path)
    from vast_upscaler.config import load_config

    loaded_settings = load_config(settings)

    callbacks = RunCallbacks(
        report_provisioning_status=lambda _payload: None,
        report_worker_progress=lambda _payload: None,
        confirm_paid=lambda _message: True,
        confirm_ssh_trust=lambda _message: True,
        request_verified_fingerprint=lambda _message: None,
        raise_if_cancelled=lambda: None,
    )

    class _FakeClient:
        def close(self) -> None:
            return

    class _NoLock:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    captured_auto: dict = {}

    def _fake_run_workflow(**kwargs):
        captured_auto["contract"] = kwargs["options"].preprocessing_contract
        return SimpleNamespace(
            session_id="sess",
            state="COMPLETED",
            worker_result=SimpleNamespace(local_output_path=output, local_report_path=report),
            estimate=SimpleNamespace(total_estimated_usd=0.1),
            instance_id=1,
            cleanup_verified=True,
            cleanup_warning=None,
        )

    monkeypatch.setattr("vast_upscaler.webapp._build_read_client", lambda _s: _FakeClient())
    monkeypatch.setattr("vast_upscaler.webapp._build_write_client", lambda _s: _FakeClient())
    monkeypatch.setattr("vast_upscaler.webapp.SessionExecutionLock", _NoLock)
    monkeypatch.setattr("vast_upscaler.webapp.run_upscale_workflow", _fake_run_workflow)
    _default_auto_runner(request, loaded_settings, callbacks)

    captured_manual: dict = {}

    class _FakeWriteClient(_FakeClient):
        def get_instance(self, _instance_id: int):
            return SimpleNamespace(
                instance_id=123,
                actual_status="running",
                gpu_name="RTX 3090",
                ssh_host="1.2.3.4",
                machine_id="node",
            )

    monkeypatch.setattr("vast_upscaler.webapp._build_write_client", lambda _s: _FakeWriteClient())
    monkeypatch.setattr("vast_upscaler.webapp._wait_for_ssh_tcp_availability", lambda **_kwargs: None)
    monkeypatch.setattr("vast_upscaler.webapp._build_trusted_connection", lambda **_kwargs: object())
    monkeypatch.setattr("vast_upscaler.webapp._resolve_private_key", lambda **_kwargs: Path("C:/keys/id_ed25519"))
    monkeypatch.setattr("vast_upscaler.webapp.OpenSshTransport.check_ssh_ready", lambda *_a, **_k: None)
    monkeypatch.setattr("vast_upscaler.webapp._run_manual_engine_preflight", lambda **_kwargs: None)

    def _fake_execute_remote_worker(**kwargs):
        captured_manual["contract"] = kwargs["request"].preprocessing_contract
        return SimpleNamespace(local_output_path=output, local_report_path=report)

    monkeypatch.setattr("vast_upscaler.webapp.execute_remote_worker", _fake_execute_remote_worker)
    manual_request = request.model_copy(update={"mode": "manual"})
    _default_manual_runner(manual_request, loaded_settings, callbacks)

    assert captured_auto["contract"] == captured_manual["contract"]
    assert captured_auto["contract"] == {
        "version": "1.0",
        "operations": [{"id": "crop", "params": {"x": 0, "y": 0, "width": 640, "height": 360}}],
    }


def test_default_auto_runner_merges_excerpt_metadata_and_cleans_temp_excerpt(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    output = tmp_path / "output.mp4"
    report = tmp_path / "report.json"
    report.write_text('{"status":"completed","report_schema_version":"2.0"}', encoding="utf-8")
    excerpt_file = tmp_path / "data" / "excerpts" / "job-excerpt-test.mkv"
    excerpt_file.parent.mkdir(parents=True, exist_ok=True)
    excerpt_file.write_bytes(b"x")

    request = StartJobRequest(
        mode="automatic",
        input_path=str(source),
        output_path=str(output),
        report_path=str(report),
        worker_image="ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
        excerpt_enabled=True,
        excerpt_start="00:01",
        excerpt_duration_seconds=2.0,
    )
    settings = _write_config(tmp_path)
    from vast_upscaler.config import load_config

    loaded_settings = load_config(settings)
    callbacks = RunCallbacks(
        report_provisioning_status=lambda _payload: None,
        report_worker_progress=lambda _payload: None,
        confirm_paid=lambda _message: True,
        confirm_ssh_trust=lambda _message: True,
        request_verified_fingerprint=lambda _message: None,
        raise_if_cancelled=lambda: None,
    )

    class _FakeClient:
        def close(self) -> None:
            return

    class _NoLock:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    effective_input = SimpleNamespace(
        input_path=excerpt_file,
        original_source=source,
        excerpt_result=SimpleNamespace(
            excerpt_path=excerpt_file,
            start_seconds=1.0,
            requested_duration_seconds=2.0,
            actual_duration_seconds=1.95,
        ),
        cleanup_excerpt_path=excerpt_file,
    )

    monkeypatch.setattr("vast_upscaler.webapp._prepare_effective_job_input", lambda **_kwargs: effective_input)
    monkeypatch.setattr("vast_upscaler.webapp._build_read_client", lambda _s: _FakeClient())
    monkeypatch.setattr("vast_upscaler.webapp._build_write_client", lambda _s: _FakeClient())
    monkeypatch.setattr("vast_upscaler.webapp.SessionExecutionLock", _NoLock)
    monkeypatch.setattr(
        "vast_upscaler.webapp.run_upscale_workflow",
        lambda **_kwargs: SimpleNamespace(
            session_id="sess",
            state="COMPLETED",
            worker_result=SimpleNamespace(local_output_path=output, local_report_path=report),
            estimate=SimpleNamespace(total_estimated_usd=0.1),
            instance_id=123,
            cleanup_verified=True,
            cleanup_warning=None,
        ),
    )

    result = _default_auto_runner(request, loaded_settings, callbacks)
    assert result.report_path == str(report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["original_source"] == str(source)
    assert payload["excerpt_enabled"] is True
    assert payload["excerpt_start_seconds"] == 1.0
    assert payload["excerpt_requested_duration_seconds"] == 2.0
    assert payload["excerpt_actual_duration_seconds"] == 1.95
    assert payload["excerpt_source"] == str(excerpt_file)
    assert not excerpt_file.exists()


def test_default_manual_seedvr2_runner_preserves_excerpt_window_and_uses_excerpt_input(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    output = tmp_path / "output.mp4"
    report = tmp_path / "report.json"
    excerpt_file = tmp_path / "data" / "excerpts" / "job-excerpt-test.mkv"
    excerpt_file.parent.mkdir(parents=True, exist_ok=True)
    excerpt_file.write_bytes(b"x")

    request = StartJobRequest(
        mode="manual",
        engine="seedvr2",
        seedvr2_profile="reference",
        input_path=str(source),
        output_path=str(output),
        report_path=str(report),
        instance_id=123,
        worker_image="ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-test",
        excerpt_enabled=True,
        excerpt_start="00:39",
        excerpt_duration_seconds=10.0,
    )

    from vast_upscaler.config import load_config

    loaded_settings = load_config(_write_config(tmp_path))

    class _FakeWriteClient:
        def get_instance(self, _instance_id: int):
            return InstanceSummary(
                instance_id=123,
                actual_status="running",
                intended_status="running",
                cur_state="running",
                label="manual-instance",
                image_runtype="ssh_direct",
                ssh_port=22,
                gpu_name="RTX 5080",
                ssh_host="1.2.3.4",
                machine_id="node",
                image_name="ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-test",
                hourly_price_usd=None,
            )

        def close(self) -> None:
            return

    captured: dict = {}

    monkeypatch.setattr("vast_upscaler.webapp._build_write_client", lambda _s: _FakeWriteClient())
    monkeypatch.setattr("vast_upscaler.webapp._wait_for_ssh_tcp_availability", lambda **_kwargs: None)
    monkeypatch.setattr("vast_upscaler.webapp._build_trusted_connection", lambda **_kwargs: object())
    monkeypatch.setattr("vast_upscaler.webapp._resolve_private_key", lambda **_kwargs: Path("C:/keys/id_ed25519"))
    monkeypatch.setattr("vast_upscaler.webapp.OpenSshTransport.check_ssh_ready", lambda *_a, **_k: None)
    monkeypatch.setattr("vast_upscaler.webapp._run_manual_engine_preflight", lambda **_kwargs: None)
    monkeypatch.setattr(
        "vast_upscaler.webapp._prepare_effective_job_input",
        lambda **_kwargs: SimpleNamespace(
            input_path=excerpt_file,
            original_source=source,
            excerpt_result=SimpleNamespace(
                excerpt_path=excerpt_file,
                start_seconds=39.0,
                requested_duration_seconds=10.0,
                actual_duration_seconds=10.0,
            ),
            cleanup_excerpt_path=excerpt_file,
        ),
    )

    def _fake_execute_remote_seedvr2_worker(**kwargs):
        captured["request"] = kwargs["request"]
        report.write_text('{"status":"completed","report_schema_version":"2.0"}', encoding="utf-8")
        output.write_bytes(b"video")
        return SimpleNamespace(local_output_path=output, local_report_path=report)

    monkeypatch.setattr("vast_upscaler.webapp.execute_remote_seedvr2_worker", _fake_execute_remote_seedvr2_worker)

    callbacks = RunCallbacks(
        report_provisioning_status=lambda _payload: None,
        report_worker_progress=lambda _payload: None,
        confirm_paid=lambda _message: True,
        confirm_ssh_trust=lambda _message: True,
        request_verified_fingerprint=lambda _message: None,
        raise_if_cancelled=lambda: None,
    )
    result = _default_manual_runner(request, loaded_settings, callbacks)
    assert result.state == "COMPLETED"
    assert captured["request"].local_input_path == excerpt_file
    assert captured["request"].profile == "reference"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["original_source"] == str(source)
    assert payload["excerpt_enabled"] is True
    assert payload["excerpt_start_seconds"] == 39.0
    assert payload["excerpt_requested_duration_seconds"] == 10.0
    assert payload["excerpt_actual_duration_seconds"] == 10.0
    assert payload["worker_image"]["requested"] == "ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-test"
    assert payload["worker_image"]["suggested_or_expected"] == "ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-test"
    assert payload["worker_image"]["instance_declared"] == "ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-test"
    assert payload["worker_image"]["verification_level"] == "exact-match"


def test_web_seedvr2_recovery_endpoint_recovers_without_reinference(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    recovery_state_path = tmp_path / "seedvr2_recovery_jobs.json"
    monkeypatch.setattr(webapp_module, "_SEEDVR2_RECOVERY_STATE_PATH", recovery_state_path)

    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"x")
    output_path = tmp_path / "output.mp4"
    report_path = tmp_path / "report.json"
    recovery_job_id = "seedvr2-job-1"
    webapp_module._upsert_seedvr2_recovery_job(
        {
            "recovery_job_id": recovery_job_id,
            "instance_id": 123,
            "engine": "seedvr2",
            "seedvr2_profile": "smoke",
            "remote_root_base": "/workspace/seedvr2-sessions",
            "remote_session_id": "manual-abc123",
            "ssh_host": "1.2.3.4",
            "ssh_port": 22,
            "local_input_path": str(input_path),
            "local_output_path": str(output_path),
            "local_report_path": str(report_path),
            "inference_state": "completed",
            "recovery_state": "pending",
        }
    )

    class _FakeWriteClient:
        def get_instance(self, _instance_id: int):
            return InstanceSummary(
                instance_id=123,
                actual_status="running",
                intended_status="running",
                cur_state="running",
                label="manual-instance",
                image_runtype="ssh_direct",
                ssh_port=22,
                gpu_name="RTX 5080",
                ssh_host="1.2.3.4",
                machine_id="node",
                image_name="ghcr.io/davbg67/vast-video-upscaler-seedvr2:exp-6a2-test",
                hourly_price_usd=None,
            )

        def close(self) -> None:
            return

    monkeypatch.setattr("vast_upscaler.webapp._build_write_client", lambda _s: _FakeWriteClient())
    monkeypatch.setattr("vast_upscaler.webapp._wait_for_ssh_tcp_availability", lambda **_kwargs: None)
    monkeypatch.setattr("vast_upscaler.webapp._build_trusted_connection", lambda **_kwargs: object())
    monkeypatch.setattr("vast_upscaler.webapp._resolve_private_key", lambda **_kwargs: Path("C:/keys/id_ed25519"))
    monkeypatch.setattr("vast_upscaler.webapp.OpenSshTransport.check_ssh_ready", lambda *_a, **_k: None)
    monkeypatch.setattr("vast_upscaler.webapp._run_manual_engine_preflight", lambda **_kwargs: None)

    def _should_not_run_inference(**_kwargs):
        raise AssertionError("inference runner must not be called during recovery-only flow")

    monkeypatch.setattr("vast_upscaler.webapp.execute_remote_seedvr2_worker", _should_not_run_inference)

    def _fake_recover_remote_seedvr2_results(**_kwargs):
        output_path.write_bytes(b"video")
        report_path.write_text('{"status":"completed","report_schema_version":"2.0"}', encoding="utf-8")
        return SimpleNamespace(local_output_path=output_path, local_report_path=report_path)

    monkeypatch.setattr("vast_upscaler.webapp.recover_remote_seedvr2_results", _fake_recover_remote_seedvr2_results)

    app = create_app(config_path=config_path)
    client = TestClient(app)
    mismatch = client.post(
        "/api/jobs/recover-seedvr2",
        json={"recovery_job_id": recovery_job_id, "instance_id": 124},
    )
    assert mismatch.status_code == 400
    assert "does not match" in mismatch.json()["detail"]

    response = client.post(
        "/api/jobs/recover-seedvr2",
        json={"recovery_job_id": recovery_job_id, "instance_id": 123},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    terminal = _wait_for_status(client, job_id, {"completed", "failed", "cancelled"})
    assert terminal["status"] == "completed"

    persisted = webapp_module._load_seedvr2_recovery_job(recovery_job_id)
    assert persisted["inference_state"] == "completed"
    assert persisted["recovery_state"] == "completed"

    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    jobs = cfg.json()["seedvr2_recovery_jobs"]
    assert any(job["recovery_job_id"] == recovery_job_id for job in jobs)


def test_web_seedvr2_recovery_jobs_persist_after_app_restart(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    recovery_state_path = tmp_path / "seedvr2_recovery_jobs.json"
    monkeypatch.setattr(webapp_module, "_SEEDVR2_RECOVERY_STATE_PATH", recovery_state_path)
    webapp_module._upsert_seedvr2_recovery_job(
        {
            "recovery_job_id": "seedvr2-persisted-job",
            "instance_id": 456,
            "engine": "seedvr2",
            "seedvr2_profile": "reference",
            "remote_root_base": "/workspace/seedvr2-sessions",
            "remote_session_id": "manual-persisted",
            "ssh_host": "1.2.3.4",
            "ssh_port": 22,
            "local_input_path": str(tmp_path / "input.mp4"),
            "local_output_path": str(tmp_path / "out.mp4"),
            "local_report_path": str(tmp_path / "report.json"),
            "inference_state": "completed",
            "recovery_state": "pending",
        }
    )

    app = create_app(config_path=config_path)
    client = TestClient(app)
    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    jobs = cfg.json()["seedvr2_recovery_jobs"]
    assert any(job["recovery_job_id"] == "seedvr2-persisted-job" for job in jobs)


def test_manual_plan_path_overrides_generated_plan_payload_for_contract_resolution(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")
    plan = tmp_path / "plan.json"
    plan.write_text(
        """
{
  "version": "1.0",
  "source_summary": {
    "file_path": "input.mp4",
    "width": 640,
    "height": 360,
    "frame_rate_fps": 24.0,
    "duration_seconds": 5.0,
    "video_codec": "h264",
    "interlace_metadata": {},
    "diagnostics_available": false
  },
  "recommendations": [],
  "user_choices": {
    "crop_mode": "custom_crop",
    "custom_crop_margins": {"top": 0, "bottom": 0, "left": 0, "right": 0},
    "deinterlace_mode": "preserve",
    "denoise_mode": "preserve",
    "deblock_mode": "preserve"
  },
  "operations": [
    {
      "operation_id": "crop",
      "kind": "crop",
      "enabled": true,
      "recommended": false,
      "confidence": "high",
      "parameters": {
        "mode": "custom_crop",
        "crop_rect": {"x": 0, "y": 0, "width": 640, "height": 360},
        "preserve_aspect_ratio": true,
        "allow_stretch": false,
        "fit_policy_after_crop": "letterbox_or_pad"
      },
      "justification": "manual",
      "risk_notes": []
    }
  ],
  "warnings": [],
  "estimated_impact": {
    "cpu_cost": "low",
    "gpu_gain": "unknown",
    "quality_risk": "low",
    "note": "n/a"
  }
}
""".strip(),
        encoding="utf-8",
    )
    request = StartJobRequest(
        mode="automatic",
        input_path=str(source),
        preprocessing_plan_path=str(plan),
        preprocessing_plan_payload={
            "version": "1.0",
            "source_summary": {
                "file_path": "input.mp4",
                "width": 640,
                "height": 360,
                "frame_rate_fps": 24.0,
                "duration_seconds": 5.0,
                "video_codec": "h264",
                "interlace_metadata": {},
                "diagnostics_available": False,
            },
            "recommendations": [],
            "user_choices": {
                "crop_mode": "preserve",
                "custom_crop_margins": None,
                "deinterlace_mode": "preserve",
                "denoise_mode": "enable",
                "deblock_mode": "preserve",
            },
            "operations": [
                {
                    "operation_id": "denoise",
                    "kind": "denoise",
                    "enabled": True,
                    "recommended": False,
                    "confidence": "low",
                    "parameters": {"mode": "enable"},
                    "justification": "manual",
                    "risk_notes": [],
                }
            ],
            "warnings": [],
            "estimated_impact": {"cpu_cost": "medium", "gpu_gain": "unknown", "quality_risk": "high", "note": "n/a"},
        },
        worker_image="ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-compat",
    )
    contract = _resolve_remote_preprocessing_contract(request=request)
    assert contract == {
        "version": "1.0",
        "operations": [{"id": "crop", "params": {"x": 0, "y": 0, "width": 640, "height": 360}}],
    }
