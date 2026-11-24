from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vast_upscaler.seedvr2_prototype import (
    PROFILE_REFERENCE,
    PROFILE_SMOKE,
    PROFILES,
    _build_cuda_preflight_command,
    _build_seedvr2_command,
    _model_identity_payload,
    detect_oom_failure,
    parse_timecode_to_seconds,
    run_cli,
)
from vast_upscaler.video import VideoProbeError


def test_seedvr2_profiles_cover_smoke_and_reference() -> None:
    assert set(PROFILES.keys()) == {"smoke", "reference"}
    assert PROFILE_SMOKE.excerpt_duration_seconds <= 3.0
    assert PROFILE_REFERENCE.target_width == 1280
    assert PROFILE_REFERENCE.target_height == 720


def test_parse_timecode_to_seconds_accepts_mmss_and_hhmmss() -> None:
    assert parse_timecode_to_seconds("00:39") == 39.0
    assert parse_timecode_to_seconds("01:02:03") == 3723.0


def test_parse_timecode_to_seconds_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        parse_timecode_to_seconds("")
    with pytest.raises(ValueError):
        parse_timecode_to_seconds("99")
    with pytest.raises(ValueError):
        parse_timecode_to_seconds("00:60")


def test_detect_oom_failure_matches_common_cuda_errors() -> None:
    assert detect_oom_failure("RuntimeError: CUDA out of memory while allocating tensor")
    assert detect_oom_failure("CUBLAS_STATUS_ALLOC_FAILED")
    assert not detect_oom_failure("Completed successfully")


def test_seedvr2_command_uses_expected_cli_flags() -> None:
    command = _build_seedvr2_command(
        excerpt_path=Path("/tmp/in.mkv"),
        raw_output_path=Path("/tmp/out.mp4"),
        profile=PROFILE_SMOKE,
        cache_dir=Path("/tmp/cache"),
        dit_model="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        attention_mode="sdpa",
    )
    rendered = " ".join(command)
    assert "--output_format mp4" in rendered
    assert "--video_backend ffmpeg" in rendered
    assert "--attention_mode sdpa" in rendered
    assert "--blocks_to_swap 32" in rendered
    assert "--vae_encode_tiled" in rendered
    assert "--vae_decode_tiled" in rendered


def test_model_identity_payload_matches_pinned_checkpoint_metadata() -> None:
    dit_metadata = _model_identity_payload("seedvr2_ema_3b_fp8_e4m3fn.safetensors")
    assert dit_metadata["repo"] == "numz/SeedVR2_comfyUI"
    assert (
        dit_metadata["download_url"]
        == "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    )
    assert (
        dit_metadata["sha256"]
        == "3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff"
    )


def test_cuda_preflight_checks_sm_120_arch() -> None:
    command = _build_cuda_preflight_command()
    assert command[0:2] == ["python", "-c"]
    assert "Missing sm_120 in torch CUDA arch list" in command[2]


@pytest.mark.parametrize(
    ("profile", "duration_seconds"),
    [
        ("smoke", 3.0),
        ("reference", 10.0),
        ("reference", 7.0),
        ("smoke", 10.0),
        ("reference", 30.0),
    ],
)
def test_run_cli_input_is_excerpt_uses_effective_input_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    duration_seconds: float,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output_root = tmp_path / "out"
    work_root = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        "vast_upscaler.seedvr2_prototype.inspect_video",
        lambda _path: SimpleNamespace(duration_seconds=duration_seconds),
    )
    exit_code = run_cli(
        [
            "--input-video",
            str(source),
            "--profile",
            profile,
            "--output-root",
            str(output_root),
            "--work-root",
            str(work_root),
            "--cache-dir",
            str(cache_dir),
            "--input-is-excerpt",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    reports = sorted(output_root.rglob("prototype-report.json"))
    assert reports, "Expected a report file in output root"
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["parameters"]["input_is_excerpt"] is True
    assert payload["parameters"]["start_seconds"] == 0.0
    assert payload["parameters"]["start_timecode"] == "00:00"
    assert payload["parameters"]["duration_seconds"] == duration_seconds
    assert "extract_excerpt" not in payload["commands"]
    assert "seedvr2_inference" in payload["commands"]


def test_run_cli_reference_profile_keeps_0039_10s_window_in_report(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output_root = tmp_path / "out"
    work_root = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    exit_code = run_cli(
        [
            "--input-video",
            str(source),
            "--profile",
            "reference",
            "--output-root",
            str(output_root),
            "--work-root",
            str(work_root),
            "--cache-dir",
            str(cache_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    reports = sorted(output_root.rglob("prototype-report.json"))
    assert reports, "Expected a report file in output root"
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["parameters"]["start_seconds"] == 39.0
    assert payload["parameters"]["duration_seconds"] == 10.0
    assert payload["parameters"]["profile_default_start_timecode"] == "00:39"
    assert "extract_excerpt" in payload["commands"]
    assert "-ss 39.000" in payload["commands"]["extract_excerpt"]


def test_run_cli_input_is_excerpt_reports_probe_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output_root = tmp_path / "out"
    work_root = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    def _raise_probe_error(_path):
        raise VideoProbeError("probe failed")

    monkeypatch.setattr("vast_upscaler.seedvr2_prototype.inspect_video", _raise_probe_error)
    exit_code = run_cli(
        [
            "--input-video",
            str(source),
            "--profile",
            "reference",
            "--output-root",
            str(output_root),
            "--work-root",
            str(work_root),
            "--cache-dir",
            str(cache_dir),
            "--input-is-excerpt",
        ]
    )
    assert exit_code == 1
    reports = sorted(output_root.rglob("prototype-report.json"))
    assert reports, "Expected a report file in output root"
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "Could not inspect input excerpt duration" in payload["error"]

