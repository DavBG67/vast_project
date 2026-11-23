from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"


@dataclass(frozen=True)
class Seedvr2PrototypeProfile:
    name: str
    excerpt_start: str
    excerpt_duration_seconds: float
    target_width: int
    target_height: int
    batch_size: int
    chunk_size: int
    blocks_to_swap: int
    vae_tile_size: int
    vae_tile_overlap: int


PROFILE_SMOKE = Seedvr2PrototypeProfile(
    name="smoke",
    excerpt_start="00:39",
    excerpt_duration_seconds=3.0,
    target_width=960,
    target_height=540,
    batch_size=5,
    chunk_size=40,
    blocks_to_swap=32,
    vae_tile_size=512,
    vae_tile_overlap=64,
)

PROFILE_REFERENCE = Seedvr2PrototypeProfile(
    name="reference",
    excerpt_start="00:39",
    excerpt_duration_seconds=10.0,
    target_width=1280,
    target_height=720,
    batch_size=5,
    chunk_size=80,
    blocks_to_swap=32,
    vae_tile_size=512,
    vae_tile_overlap=64,
)

PROFILES = {
    PROFILE_SMOKE.name: PROFILE_SMOKE,
    PROFILE_REFERENCE.name: PROFILE_REFERENCE,
}

_OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "cuda error: memory allocation",
)


def parse_timecode_to_seconds(value: str) -> float:
    token = value.strip()
    if not token:
        raise ValueError("Timecode cannot be empty.")
    parts = token.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid timecode '{value}'. Expected MM:SS or HH:MM:SS.")
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid timecode '{value}'.")
    if len(parts) == 2:
        minutes, seconds = (int(parts[0]), int(parts[1]))
        hours = 0
    else:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    if minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid timecode '{value}'.")
    return float((hours * 3600) + (minutes * 60) + seconds)


def detect_oom_failure(log_output: str) -> bool:
    lowered = log_output.lower()
    return any(pattern in lowered for pattern in _OOM_PATTERNS)


def _build_extract_command(
    input_video: Path,
    excerpt_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(input_video),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(excerpt_path),
    ]


def _build_seedvr2_command(
    excerpt_path: Path,
    raw_output_path: Path,
    profile: Seedvr2PrototypeProfile,
    cache_dir: Path,
    dit_model: str,
    attention_mode: str,
) -> list[str]:
    resolution = min(profile.target_width, profile.target_height)
    max_resolution = max(profile.target_width, profile.target_height)
    return [
        "python",
        "/opt/seedvr2/inference_cli.py",
        str(excerpt_path),
        "--output",
        str(raw_output_path),
        "--output_format",
        "mp4",
        "--video_backend",
        "ffmpeg",
        "--dit_model",
        dit_model,
        "--model_dir",
        str(cache_dir),
        "--attention_mode",
        attention_mode,
        "--resolution",
        str(resolution),
        "--max_resolution",
        str(max_resolution),
        "--batch_size",
        str(profile.batch_size),
        "--uniform_batch_size",
        "--chunk_size",
        str(profile.chunk_size),
        "--blocks_to_swap",
        str(profile.blocks_to_swap),
        "--swap_io_components",
        "--dit_offload_device",
        "cpu",
        "--vae_offload_device",
        "cpu",
        "--tensor_offload_device",
        "cpu",
        "--vae_encode_tiled",
        "--vae_decode_tiled",
        "--vae_encode_tile_size",
        str(profile.vae_tile_size),
        "--vae_decode_tile_size",
        str(profile.vae_tile_size),
        "--vae_encode_tile_overlap",
        str(profile.vae_tile_overlap),
        "--vae_decode_tile_overlap",
        str(profile.vae_tile_overlap),
        "--debug",
    ]


def _build_mux_command(raw_output_path: Path, excerpt_path: Path, final_output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_output_path),
        "-i",
        str(excerpt_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(final_output_path),
    ]


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _run_command(command: Sequence[str], log_file, stage_name: str) -> tuple[int, str, float]:
    command_text = _format_command(command)
    log_file.write(f"\n[{stage_name}] COMMAND\n{command_text}\n")
    log_file.flush()
    stage_start = time.time()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
    return_code = process.wait()
    elapsed = time.time() - stage_start
    output_text = "".join(output_lines)
    log_file.write(f"[{stage_name}] EXIT={return_code} DURATION_SECONDS={elapsed:.3f}\n")
    log_file.flush()
    return (return_code, output_text, elapsed)


def _safe_name(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "video"


def _write_report(report_path: Path, payload: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SeedVR2 standalone Docker prototype runner")
    parser.add_argument("--input-video", required=True, help="Absolute path of source video inside the container")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="smoke")
    parser.add_argument("--output-root", default="/workspace/output")
    parser.add_argument("--work-root", default="/workspace/tmp")
    parser.add_argument("--cache-dir", default="/workspace/cache/seedvr2")
    parser.add_argument("--start-timecode", default=None, help="Override profile excerpt start (MM:SS or HH:MM:SS)")
    parser.add_argument("--duration-seconds", type=float, default=None, help="Override profile excerpt duration")
    parser.add_argument("--dit-model", default=DEFAULT_DIT_MODEL)
    parser.add_argument("--attention-mode", default="sdpa", choices=("sdpa",))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_video = Path(args.input_video).resolve()
    output_root = Path(args.output_root).resolve()
    work_root = Path(args.work_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    profile = PROFILES[args.profile]

    run_id = time.strftime("%Y%m%d-%H%M%S")
    source_stem = _safe_name(input_video.stem)
    run_dir = output_root / f"seedvr2-{profile.name}-{run_id}-{source_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_root / run_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_timecode = args.start_timecode or profile.excerpt_start
    start_seconds = parse_timecode_to_seconds(start_timecode)
    duration_seconds = float(args.duration_seconds or profile.excerpt_duration_seconds)

    excerpt_path = work_dir / "input_excerpt.mkv"
    raw_output_path = work_dir / "seedvr2_raw.mp4"
    final_output_path = run_dir / "upscaled_h264_aac.mp4"
    log_path = run_dir / "run.log"
    report_path = run_dir / "prototype-report.json"

    commands = {
        "extract_excerpt": _build_extract_command(
            input_video=input_video,
            excerpt_path=excerpt_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        ),
        "seedvr2_inference": _build_seedvr2_command(
            excerpt_path=excerpt_path,
            raw_output_path=raw_output_path,
            profile=profile,
            cache_dir=cache_dir,
            dit_model=args.dit_model,
            attention_mode=args.attention_mode,
        ),
        "mux_h264_aac": _build_mux_command(
            raw_output_path=raw_output_path,
            excerpt_path=excerpt_path,
            final_output_path=final_output_path,
        ),
    }

    report_payload = {
        "status": "pending",
        "profile": profile.name,
        "input_video": str(input_video),
        "run_directory": str(run_dir),
        "work_directory": str(work_dir),
        "cache_directory": str(cache_dir),
        "checkpoints": {
            "dit_model": args.dit_model,
            "vae_model": DEFAULT_VAE_MODEL,
        },
        "attention_mode": args.attention_mode,
        "parameters": {
            "profile": asdict(profile),
            "start_timecode": start_timecode,
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
        },
        "commands": {key: _format_command(value) for key, value in commands.items()},
        "log_file": str(log_path),
        "output_file": str(final_output_path),
        "oom_detected": False,
    }

    if not input_video.exists():
        report_payload["status"] = "failed"
        report_payload["error"] = f"Input video does not exist: {input_video}"
        _write_report(report_path, report_payload)
        print(report_payload["error"], file=sys.stderr)
        return 1

    if args.dry_run:
        report_payload["status"] = "dry-run"
        report_payload["dry_run"] = True
        _write_report(report_path, report_payload)
        print(json.dumps(report_payload, indent=2))
        return 0

    stage_timings: dict[str, float] = {}
    started_at = time.time()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"seedvr2_prototype_run={run_id}\n")
        log_file.write(json.dumps(report_payload["parameters"], indent=2))
        log_file.write("\n")
        log_file.flush()

        for stage_name, command in commands.items():
            return_code, stage_output, duration = _run_command(command, log_file=log_file, stage_name=stage_name)
            stage_timings[stage_name] = duration
            if return_code != 0:
                report_payload["status"] = "failed"
                report_payload["failed_stage"] = stage_name
                report_payload["failed_exit_code"] = return_code
                report_payload["oom_detected"] = detect_oom_failure(stage_output)
                report_payload["timings_seconds"] = stage_timings
                report_payload["elapsed_seconds"] = time.time() - started_at
                _write_report(report_path, report_payload)
                return 2 if report_payload["oom_detected"] else 1

    report_payload["status"] = "completed"
    report_payload["timings_seconds"] = stage_timings
    report_payload["elapsed_seconds"] = time.time() - started_at
    _write_report(report_path, report_payload)
    print(f"[seedvr2-prototype] completed: {final_output_path}")
    return 0

