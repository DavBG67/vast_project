from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    from .models import ProfileNotSupportedError, select_model
    from .pipelines.realesrgan import (
        RealesrganConfig,
        RealesrganError,
        RealesrganUpscaler,
    )
    from .video_probe import ProbeError, probe_video
except ImportError:
    from models import ProfileNotSupportedError, select_model
    from pipelines.realesrgan import (
        RealesrganConfig,
        RealesrganError,
        RealesrganUpscaler,
    )
    from video_probe import ProbeError, probe_video


class WorkerError(RuntimeError):
    pass


class ProcessingStageError(WorkerError):
    def __init__(self, message: str, frames_processed: int) -> None:
        super().__init__(message)
        self.frames_processed = frames_processed


_STOP_REQUESTED = False
PROGRESS_EVENT_PREFIX = "VAST_PROGRESS_JSON "
PROGRESS_EVENT_VERSION = 1


def _signal_handler(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


@dataclass(frozen=True)
class ProcessingReport:
    status: str
    source: str
    output: str
    model: str
    profile: str
    frames_processed: int
    processing_seconds: float
    processing_fps: float
    gpu: str
    notes: list[str]
    preprocessing_plan: dict | None = None
    comparison_context: dict | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    summary: dict
    crop_rect: tuple[int, int, int, int] | None
    decode_filter: str | None
    source_width_before: int
    source_height_before: int
    source_width_effective: int
    source_height_effective: int


class ProgressEmitter:
    def __init__(self, *, started_at: float, emit_interval_seconds: float = 1.0, emit_frame_interval: int = 25) -> None:
        self._started_at = started_at
        self._emit_interval_seconds = emit_interval_seconds
        self._emit_frame_interval = emit_frame_interval
        self._last_emitted_at = 0.0
        self._last_emitted_frames = -1
        self._last_stage: str | None = None

    def emit(
        self,
        *,
        stage: str,
        frames_done: int | None = None,
        frames_total: int | None = None,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if not force:
            enough_time = (now - self._last_emitted_at) >= self._emit_interval_seconds
            enough_frames = (
                frames_done is not None
                and frames_done >= 0
                and (self._last_emitted_frames < 0 or frames_done - self._last_emitted_frames >= self._emit_frame_interval)
            )
            stage_changed = stage != self._last_stage
            if not (enough_time or enough_frames or stage_changed):
                return

        elapsed = max(0.0, now - self._started_at)
        fps = (frames_done / elapsed) if frames_done is not None and elapsed > 0 else None
        eta: float | None = None
        if frames_done is not None and frames_total is not None and fps and fps > 0 and frames_total >= frames_done:
            eta = (frames_total - frames_done) / fps

        payload: dict[str, object] = {
            "kind": "worker",
            "event_version": PROGRESS_EVENT_VERSION,
            "stage": stage,
            "elapsed_seconds": round(elapsed, 3),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if frames_done is not None:
            payload["frames_done"] = max(0, frames_done)
        if frames_total is not None:
            payload["frames_total"] = max(0, frames_total)
        if fps is not None:
            payload["fps_processing"] = round(fps, 3)
        if eta is not None:
            payload["eta_seconds"] = max(0.0, round(eta, 3))

        print(f"{PROGRESS_EVENT_PREFIX}{json.dumps(payload, separators=(',', ':'))}", flush=True)
        self._last_stage = stage
        self._last_emitted_at = now
        if frames_done is not None:
            self._last_emitted_frames = frames_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vast Video Upscaler worker (phase 3)")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--video-crf", type=int, default=20)
    parser.add_argument("--models-dir", default="/opt/models")
    parser.add_argument("--report", default=None, help="JSON report path")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--preprocessing-plan", default=None, help="Remote preprocessing contract JSON path")
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output_path.with_suffix(output_path.suffix + ".report.json")

    started = time.perf_counter()
    progress = ProgressEmitter(started_at=started)
    notes: list[str] = []
    preprocessing_plan_summary: dict | None = None
    frames_processed = 0
    staged_output: Path | None = None

    try:
        _emit_cuda_diagnostics()
        progress.emit(stage="preparation", frames_done=0, force=True)
        if not input_path.exists():
            raise WorkerError(f"Input file does not exist: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_disk_space(output_path.parent, min_free_gb=args.min_free_gb)

        source = probe_video(input_path)
        preprocessing_config = PreprocessingConfig(
            summary={
                "present": False,
                "received": False,
                "validated": False,
                "version": None,
                "requested_operations": [],
                "applied_operations": [],
                "source_dimensions": {"width": source.width, "height": source.height},
                "effective_dimensions": {"width": source.width, "height": source.height},
                "decode_filter": None,
            },
            crop_rect=None,
            decode_filter=None,
            source_width_before=source.width,
            source_height_before=source.height,
            source_width_effective=source.width,
            source_height_effective=source.height,
        )
        if args.preprocessing_plan:
            preprocessing_config = _load_and_validate_preprocessing_plan(
                plan_path=Path(args.preprocessing_plan).resolve(),
                source_width=source.width,
                source_height=source.height,
            )
            preprocessing_plan_summary = preprocessing_config.summary
        source_width_for_upscale = preprocessing_config.source_width_effective
        source_height_for_upscale = preprocessing_config.source_height_effective
        frames_total = _estimate_total_frames(duration_seconds=source.duration_seconds, fps=source.fps)
        target_width, target_height = compute_target_canvas(
            source_width=source.width,
            source_height=source.height,
            target_width=args.target_width,
            target_height=args.target_height,
        )

        model = select_model(profile=args.profile, models_dir=Path(args.models_dir))
        upscaler = RealesrganUpscaler(
            RealesrganConfig(
                weights_path=model.weights_path,
                model_name=model.model_name,
                model_scale=model.model_scale,
                tile=model.tile,
                tile_pad=model.tile_pad,
                pre_pad=model.pre_pad,
            )
        )
        upscaler.load()

        with tempfile.TemporaryDirectory(prefix="vast_worker_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            encoded_video = temp_dir_path / "video_only.mkv"
            staged_output = _build_staged_output_path(output_path)

            frames_processed = run_streaming_upscale(
                input_path=input_path,
                encoded_video_path=encoded_video,
                source_width=source_width_for_upscale,
                source_height=source_height_for_upscale,
                source_fps=source.fps,
                target_width=target_width,
                target_height=target_height,
                upscaler=upscaler,
                crf=args.video_crf,
                decode_filter=preprocessing_config.decode_filter,
                progress=progress,
                frames_total=frames_total,
            )
            if preprocessing_plan_summary is not None and preprocessing_config.crop_rect is not None:
                preprocessing_plan_summary["applied_operations"] = ["crop"]

            progress.emit(stage="finalization", frames_done=frames_processed, frames_total=frames_total, force=True)
            mux_with_source_streams(input_path=input_path, video_only_path=encoded_video, output_path=staged_output)
            validate_output(output_path=staged_output, source_duration=source.duration_seconds, source_fps=source.fps)

            if output_path.exists():
                output_path.unlink()
            staged_output.rename(output_path)

            elapsed = time.perf_counter() - started
            fps = (frames_processed / elapsed) if elapsed > 0 else 0.0
            report = ProcessingReport(
                status="completed",
                source=str(input_path),
                output=str(output_path),
                model=model.name,
                profile=args.profile,
                frames_processed=frames_processed,
                processing_seconds=elapsed,
                processing_fps=fps,
                gpu=RealesrganUpscaler.detect_gpu_name(),
                notes=notes,
                preprocessing_plan=preprocessing_plan_summary,
                comparison_context={
                    "resolutions": {
                        "source": {"width": source.width, "height": source.height},
                        "effective_after_preprocessing": {
                            "width": source_width_for_upscale,
                            "height": source_height_for_upscale,
                        },
                        "target_canvas": {"width": target_width, "height": target_height},
                    },
                    "upscaler": {
                        "engine": "realesrgan",
                        "model": model.model_name,
                        "profile": args.profile,
                        "model_scale": model.model_scale,
                        "tile": model.tile,
                        "tile_pad": model.tile_pad,
                        "pre_pad": model.pre_pad,
                    },
                    "encoding": {
                        "container": output_path.suffix.lstrip(".") or None,
                        "video_codec": "libx265",
                        "preset": "medium",
                        "crf": args.video_crf,
                        "pixel_format": "yuv420p",
                    },
                },
            )
            write_report(report_path, report)
            progress.emit(stage="completed", frames_done=frames_processed, frames_total=frames_total, force=True)

        print(f"Completed: {output_path}")
        return 0

    except ProfileNotSupportedError as exc:
        _cleanup_staged_output(staged_output)
        return _fail(
            report_path,
            args,
            started,
            notes + [str(exc)],
            status="failed",
            frames_processed=frames_processed,
            preprocessing_plan=preprocessing_plan_summary,
        )
    except ProcessingStageError as exc:
        _cleanup_staged_output(staged_output)
        frames_processed = max(frames_processed, exc.frames_processed)
        return _fail(
            report_path,
            args,
            started,
            notes + [str(exc)],
            status="failed",
            frames_processed=frames_processed,
            preprocessing_plan=preprocessing_plan_summary,
        )
    except (WorkerError, ProbeError, RealesrganError) as exc:
        _cleanup_staged_output(staged_output)
        return _fail(
            report_path,
            args,
            started,
            notes + [str(exc)],
            status="failed",
            frames_processed=frames_processed,
            preprocessing_plan=preprocessing_plan_summary,
        )
    except Exception as exc:  # noqa: BLE001
        _cleanup_staged_output(staged_output)
        return _fail(
            report_path,
            args,
            started,
            notes + [f"Unexpected error: {exc}"],
            status="failed",
            frames_processed=frames_processed,
            preprocessing_plan=preprocessing_plan_summary,
        )


def run_streaming_upscale(
    input_path: Path,
    encoded_video_path: Path,
    source_width: int,
    source_height: int,
    source_fps: float,
    target_width: int,
    target_height: int,
    upscaler: RealesrganUpscaler,
    crf: int,
    decode_filter: str | None,
    progress: ProgressEmitter,
    frames_total: int | None,
) -> int:
    import cv2
    import numpy as np

    decode_cmd = _build_decode_command(input_path=input_path, decode_filter=decode_filter)

    encode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{target_width}x{target_height}",
        "-r",
        f"{source_fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(encoded_video_path),
    ]

    frame_size = source_width * source_height * 3
    frames = 0

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        progress.emit(stage="decoding", frames_done=0, frames_total=frames_total, force=True)
        while True:
            if _STOP_REQUESTED:
                raise WorkerError("Stop requested by signal")

            assert decoder.stdout is not None
            chunk = decoder.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise WorkerError("Decoder returned an incomplete frame")

            rgb = np.frombuffer(chunk, dtype=np.uint8).reshape((source_height, source_width, 3))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            upscaled_bgr = upscaler.upscale_frame(bgr)
            fitted_bgr = fit_with_letterbox(upscaled_bgr, target_width=target_width, target_height=target_height)
            out_rgb = cv2.cvtColor(fitted_bgr, cv2.COLOR_BGR2RGB)

            assert encoder.stdin is not None
            encoder.stdin.write(out_rgb.tobytes())
            frames += 1
            progress.emit(stage="upscale", frames_done=frames, frames_total=frames_total)

        assert encoder.stdin is not None
        encoder.stdin.close()

        progress.emit(stage="encoding", frames_done=frames, frames_total=frames_total, force=True)
        decode_code = decoder.wait()
        encode_code = encoder.wait()

        if decode_code != 0:
            stderr = _safe_decode(decoder.stderr)
            raise ProcessingStageError(f"FFmpeg decode failed: {stderr}", frames_processed=frames)
        if encode_code != 0:
            stderr = _safe_decode(encoder.stderr)
            raise ProcessingStageError(f"FFmpeg encode failed: {stderr}", frames_processed=frames)

        if frames == 0:
            raise WorkerError("No frames were processed")

        return frames
    finally:
        _terminate_process(decoder)
        _terminate_process(encoder)


def fit_with_letterbox(image_bgr, target_width: int, target_height: int):
    import cv2
    import numpy as np

    src_h, src_w = image_bgr.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise WorkerError("Invalid frame dimensions")

    scale = min(target_width / src_w, target_height / src_h)
    resized_w = max(1, round(src_w * scale))
    resized_h = max(1, round(src_h * scale))

    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x0 = (target_width - resized_w) // 2
    y0 = (target_height - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def compute_target_canvas(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise WorkerError("Invalid source dimensions")
    if target_width <= 0 or target_height <= 0:
        raise WorkerError("Invalid target dimensions")
    return target_width, target_height


def _build_decode_command(*, input_path: Path, decode_filter: str | None) -> list[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
    ]
    if decode_filter:
        cmd.extend(["-vf", decode_filter])
    cmd.extend(
        [
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-vsync",
            "0",
            "-",
        ]
    )
    return cmd


def mux_with_source_streams(input_path: Path, video_only_path: Path, output_path: Path) -> None:
    cmd_primary = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "copy",
        str(output_path),
    ]

    result = subprocess.run(cmd_primary, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return

    cmd_fallback = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        str(output_path),
    ]
    result_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True, check=False)
    if result_fallback.returncode != 0:
        raise WorkerError(result_fallback.stderr.strip() or "Failed to mux output streams")


def _build_staged_output_path(output_path: Path) -> Path:
    suffix = output_path.suffix
    if suffix:
        return output_path.with_name(f"{output_path.stem}.tmp{suffix}")
    return output_path.with_name(f"{output_path.name}.tmp")


def _cleanup_staged_output(staged_output: Path | None) -> None:
    if staged_output is None:
        return
    try:
        if staged_output.exists():
            staged_output.unlink()
    except OSError:
        # Keep original failure reason; temp cleanup is best effort.
        pass


def validate_output(output_path: Path, source_duration: float | None, source_fps: float) -> None:
    if not output_path.exists():
        raise WorkerError("Output file was not produced")
    if output_path.stat().st_size <= 0:
        raise WorkerError("Output file is empty")

    out_probe = probe_video(output_path)
    if out_probe.width <= 0 or out_probe.height <= 0:
        raise WorkerError("Output has invalid resolution")

    fps_delta = abs(out_probe.fps - source_fps)
    if fps_delta > 0.2:
        raise WorkerError(f"Output frame rate mismatch too high: {out_probe.fps} vs {source_fps}")

    if source_duration is not None and out_probe.duration_seconds is not None:
        tol = max(1.0, source_duration * 0.02)
        if abs(out_probe.duration_seconds - source_duration) > tol:
            raise WorkerError(
                f"Output duration mismatch too high: {out_probe.duration_seconds:.2f}s vs {source_duration:.2f}s"
            )


def write_report(path: Path, report: ProcessingReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": report.status,
        "source": report.source,
        "output": report.output,
        "model": report.model,
        "profile": report.profile,
        "frames_processed": report.frames_processed,
        "processing_seconds": round(report.processing_seconds, 3),
        "processing_fps": round(report.processing_fps, 3),
        "gpu": report.gpu,
        "notes": report.notes,
    }
    if report.preprocessing_plan is not None:
        payload["preprocessing_plan"] = report.preprocessing_plan
    payload["report_schema_version"] = "2.0"
    payload["instance_metadata"] = {
        "instance_id": None,
        "offer_id": None,
        "host_id": None,
        "sensitive_identifiers_included": False,
    }
    payload["processing_context"] = report.comparison_context or {}
    payload["telemetry"] = {
        "sampling_interval_seconds": None,
        "raw_samples": [],
        "aggregates": {},
    }
    payload["transfer_metrics"] = {
        "upload": {"duration_seconds": None, "bytes_transferred": None, "average_bytes_per_second": None},
        "download": {"duration_seconds": None, "bytes_transferred": None, "average_bytes_per_second": None},
    }
    payload["timings"] = {
        "processing_seconds": round(report.processing_seconds, 3),
        "upload_seconds": None,
        "download_seconds": None,
        "total_job_seconds": None,
    }
    payload["cost_estimate"] = {
        "currency": "USD",
        "gpu_processing_cost_estimated": None,
        "total_job_cost_estimated": None,
        "note": "Estimated cost fields are populated in orchestration/report-enrichment phases.",
    }
    payload["metric_schema"] = {
        "provenance_values": ["observed", "declared", "estimated"],
        "units": {
            "duration": "seconds",
            "throughput": "bytes/second",
            "data_size": "bytes",
            "frame_rate": "frames/second",
            "temperature": "celsius",
            "power": "watts",
            "frequency": "MHz",
            "memory": "bytes",
            "cost": "USD",
        },
    }
    payload["collection_warnings"] = []
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fail(
    report_path: Path,
    args: argparse.Namespace,
    started: float,
    notes: list[str],
    status: str,
    frames_processed: int = 0,
    preprocessing_plan: dict | None = None,
) -> int:
    elapsed = time.perf_counter() - started
    report = ProcessingReport(
        status=status,
        source=str(args.input),
        output=str(args.output),
        model="realesrgan",
        profile=str(args.profile),
        frames_processed=max(0, frames_processed),
        processing_seconds=elapsed,
        processing_fps=(max(0, frames_processed) / elapsed) if elapsed > 0 else 0.0,
        gpu=RealesrganUpscaler.detect_gpu_name(),
        notes=notes,
        preprocessing_plan=preprocessing_plan,
    )
    write_report(report_path, report)
    for note in notes:
        print(note)
    return 1


def _load_and_validate_preprocessing_plan(
    *,
    plan_path: Path,
    source_width: int,
    source_height: int,
) -> PreprocessingConfig:
    if not plan_path.exists() or not plan_path.is_file():
        raise WorkerError(f"Preprocessing plan file does not exist: {plan_path}")
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkerError(f"Unable to read preprocessing plan file: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkerError(f"Invalid preprocessing plan JSON file: {plan_path}") from exc
    if not isinstance(payload, dict):
        raise WorkerError("Preprocessing plan root must be a JSON object.")

    version = payload.get("version")
    if version != "1.0":
        raise WorkerError(f"Unsupported preprocessing plan contract version: {version!r}. Expected '1.0'.")

    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise WorkerError("Preprocessing plan must contain an operations array.")

    requested_operations: list[str] = []
    seen_ids: set[str] = set()
    crop_rect: tuple[int, int, int, int] | None = None
    for item in operations:
        if not isinstance(item, dict):
            raise WorkerError("Each preprocessing operation must be a JSON object.")
        operation_id = item.get("id")
        if operation_id not in {"crop", "deinterlace"}:
            raise WorkerError(f"Unsupported preprocessing operation id: {operation_id!r}")
        if operation_id in seen_ids:
            raise WorkerError(f"Duplicate preprocessing operation id: {operation_id!r}")
        seen_ids.add(operation_id)

        params = item.get("params")
        if not isinstance(params, dict):
            raise WorkerError(f"Operation '{operation_id}' must provide params as a JSON object.")
        if operation_id == "crop":
            _validate_crop_operation_params(params=params, source_width=source_width, source_height=source_height)
            if crop_rect is not None:
                raise WorkerError("Duplicate preprocessing crop operation is not allowed.")
            crop_rect = (int(params["x"]), int(params["y"]), int(params["width"]), int(params["height"]))
        elif operation_id == "deinterlace":
            _validate_deinterlace_operation_params(params=params)
            raise WorkerError("Deinterlace operation is not supported in this phase. Remove it from the contract.")
        requested_operations.append(str(operation_id))

    effective_width = source_width
    effective_height = source_height
    decode_filter = None
    if crop_rect is not None:
        _, _, crop_width, crop_height = crop_rect
        effective_width = crop_width
        effective_height = crop_height
        decode_filter = f"crop={crop_width}:{crop_height}:{crop_rect[0]}:{crop_rect[1]}"

    return PreprocessingConfig(
        summary={
            "present": True,
            "received": True,
            "validated": True,
            "version": "1.0",
            "requested_operations": requested_operations,
            "applied_operations": [],
            "source_dimensions": {"width": source_width, "height": source_height},
            "effective_dimensions": {"width": effective_width, "height": effective_height},
            "decode_filter": decode_filter,
        },
        crop_rect=crop_rect,
        decode_filter=decode_filter,
        source_width_before=source_width,
        source_height_before=source_height,
        source_width_effective=effective_width,
        source_height_effective=effective_height,
    )


def _validate_crop_operation_params(*, params: dict, source_width: int, source_height: int) -> None:
    required_keys = {"x", "y", "width", "height"}
    missing = sorted(required_keys.difference(params.keys()))
    if missing:
        raise WorkerError(f"Crop operation params missing required keys: {', '.join(missing)}")
    x = params.get("x")
    y = params.get("y")
    width = params.get("width")
    height = params.get("height")
    if not all(isinstance(value, int) for value in (x, y, width, height)):
        raise WorkerError("Crop operation params x/y/width/height must be integers.")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise WorkerError("Crop operation params must define positive in-bounds coordinates and dimensions.")
    if x + width > source_width or y + height > source_height:
        raise WorkerError("Crop operation rectangle is out of source bounds.")
    if width % 2 != 0 or height % 2 != 0 or x % 2 != 0 or y % 2 != 0:
        raise WorkerError("Crop operation params must use even x/y/width/height for yuv420 compatibility.")


def _validate_deinterlace_operation_params(*, params: dict) -> None:
    if params.get("algorithm") != "bwdif":
        raise WorkerError("Deinterlace operation algorithm must be 'bwdif'.")
    if params.get("mode") != "send_frame":
        raise WorkerError("Deinterlace operation mode must be 'send_frame'.")
    if params.get("parity") != "auto":
        raise WorkerError("Deinterlace operation parity must be 'auto'.")
    if params.get("deint") != "all":
        raise WorkerError("Deinterlace operation deint must be 'all'.")


def _ensure_disk_space(directory: Path, min_free_gb: float) -> None:
    usage = shutil.disk_usage(directory)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise WorkerError(f"Insufficient disk space in {directory}: {free_gb:.2f} GB available, need {min_free_gb:.2f} GB")


def _emit_cuda_diagnostics() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"[cuda-diagnostics] unavailable: unable to import torch ({exc})", file=sys.stderr, flush=True)
        return

    torch_version = str(getattr(torch, "__version__", "unknown"))
    compiled_cuda = str(getattr(torch.version, "cuda", None))
    cuda_available = bool(torch.cuda.is_available())

    gpu_name = "unavailable"
    capability_text = "n/a"
    if cuda_available:
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:  # noqa: BLE001
            gpu_name = "unknown"
        try:
            capability = torch.cuda.get_device_capability(0)
            capability_text = f"{capability[0]}.{capability[1]}"
        except Exception:  # noqa: BLE001
            capability_text = "unknown"

    arch_list_raw: list[str] | None = None
    try:
        candidate = torch.cuda.get_arch_list()
        if isinstance(candidate, list):
            arch_list_raw = [str(item) for item in candidate]
    except Exception:  # noqa: BLE001
        arch_list_raw = None

    arch_text = ",".join(arch_list_raw) if arch_list_raw else "unavailable"
    print(
        (
            f"[cuda-diagnostics] torch={torch_version} compiled_cuda={compiled_cuda} "
            f"cuda_available={cuda_available} gpu={gpu_name} capability={capability_text} arch_list={arch_text}"
        ),
        file=sys.stderr,
        flush=True,
    )

    if cuda_available and arch_list_raw is not None:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            required_sm = f"sm_{major}{minor}"
            required_compute = f"compute_{major}{minor}"
        except Exception:  # noqa: BLE001
            return
        if required_sm not in arch_list_raw and required_compute not in arch_list_raw:
            raise WorkerError(
                "CUDA runtime appears incompatible with detected GPU architecture. "
                f"GPU capability={major}.{minor}, required arch token {required_sm} not present in torch arch list {arch_list_raw}."
            )


def _estimate_total_frames(*, duration_seconds: float | None, fps: float) -> int | None:
    if duration_seconds is None or duration_seconds <= 0 or fps <= 0:
        return None
    estimate = round(duration_seconds * fps)
    return estimate if estimate > 0 else None


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def _safe_decode(stream) -> str:
    if stream is None:
        return ""
    content = stream.read()
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace").strip()
    if isinstance(content, str):
        return content.strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
