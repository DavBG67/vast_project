from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vast_upscaler.preprocessing_planner import (
    CropRect,
    PreprocessingPlan,
)
from vast_upscaler.video import VideoProbeError, inspect_video, run_ffprobe


class LocalPreprocessingError(RuntimeError):
    pass


class LocalPreprocessingValidationError(LocalPreprocessingError):
    pass


class LocalPreprocessingExecutionError(LocalPreprocessingError):
    pass


class LocalEncodingOptions(BaseModel):
    video_codec: str = "libx264"
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    copy_audio: bool = True
    copy_subtitles: bool = True


class LocalExecutionOptions(BaseModel):
    dry_run: bool = False
    overwrite_output: bool = False
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    timeout_seconds: float = Field(default=1800.0, gt=1.0, le=86_400.0)


class LocalExecutionReport(BaseModel):
    version: str = "1.0"
    status: Literal["dry_run", "completed"]
    source_path: str
    destination_path: str
    planned_operations: list[str]
    applied_operations: list[str]
    filter_chain: str
    ffmpeg_command: list[str]
    encoding: dict
    resolution_before: dict
    resolution_after: dict | None
    duration_seconds: float
    warnings: list[str]
    ffprobe_validation: dict | None


def load_preprocessing_plan(path: Path) -> PreprocessingPlan:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LocalPreprocessingError(f"Unable to read preprocessing plan: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LocalPreprocessingError(f"Invalid preprocessing plan JSON: {path}") from exc
    try:
        return PreprocessingPlan.model_validate(payload)
    except Exception as exc:
        raise LocalPreprocessingValidationError(f"Invalid preprocessing plan structure: {exc}") from exc


def execute_preprocessing_plan(
    *,
    plan: PreprocessingPlan,
    output_path: Path,
    source_path: Path | None = None,
    encoding: LocalEncodingOptions | None = None,
    options: LocalExecutionOptions | None = None,
) -> LocalExecutionReport:
    enc = encoding or LocalEncodingOptions()
    run_options = options or LocalExecutionOptions()
    source = (source_path or Path(plan.source_summary.file_path)).resolve()
    destination = output_path.resolve()

    if source == destination:
        raise LocalPreprocessingValidationError("Source and destination must be different.")
    if not source.exists() or not source.is_file():
        raise LocalPreprocessingValidationError(f"Source video does not exist: {source}")
    if destination.exists() and not run_options.overwrite_output:
        raise LocalPreprocessingValidationError(f"Destination already exists: {destination}")
    if not destination.parent.exists():
        raise LocalPreprocessingValidationError(f"Destination directory does not exist: {destination.parent}")

    filter_parts, applied_operations, warnings, expected_dims = _build_filter_chain(plan)
    if not applied_operations:
        raise LocalPreprocessingValidationError("No preprocessing operation is explicitly enabled in the plan.")

    filter_chain = ",".join(filter_parts)
    planned_operations = [op.operation_id for op in plan.operations if op.enabled]
    ffmpeg_command = _build_ffmpeg_command(
        source=source,
        destination=destination,
        filter_chain=filter_chain,
        encoding=enc,
        ffmpeg_bin=run_options.ffmpeg_bin,
    )
    before_info = inspect_video(source, ffprobe_bin=run_options.ffprobe_bin)
    resolution_before = {"width": before_info.width, "height": before_info.height}

    if run_options.dry_run:
        return LocalExecutionReport(
            status="dry_run",
            source_path=str(source),
            destination_path=str(destination),
            planned_operations=planned_operations,
            applied_operations=applied_operations,
            filter_chain=filter_chain,
            ffmpeg_command=ffmpeg_command,
            encoding=enc.model_dump(mode="json"),
            resolution_before=resolution_before,
            resolution_after={"width": expected_dims[0], "height": expected_dims[1]} if expected_dims is not None else None,
            duration_seconds=0.0,
            warnings=warnings,
            ffprobe_validation=None,
        )

    temp_output = _build_temp_output_path(destination)
    started = time.perf_counter()
    try:
        run = subprocess.run(
            ffmpeg_command[:-1] + [str(temp_output)],
            capture_output=True,
            text=True,
            check=False,
            timeout=run_options.timeout_seconds,
        )
        if run.returncode != 0:
            raise LocalPreprocessingExecutionError(
                "FFmpeg preprocessing failed. "
                f"Command exited with code {run.returncode}. stderr={run.stderr.strip() or 'n/a'}"
            )

        after_info = inspect_video(temp_output, ffprobe_bin=run_options.ffprobe_bin)
        _validate_post_output(
            source_info=before_info,
            output_info=after_info,
            expected_dims=expected_dims,
        )
        ffprobe_payload = run_ffprobe(video_path=temp_output, ffprobe_bin=run_options.ffprobe_bin)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and run_options.overwrite_output:
            destination.unlink()
        temp_output.replace(destination)
    except (OSError, subprocess.TimeoutExpired, VideoProbeError) as exc:
        _cleanup_temp_file(temp_output)
        if isinstance(exc, LocalPreprocessingExecutionError):
            raise
        raise LocalPreprocessingExecutionError(str(exc)) from exc
    except Exception:
        _cleanup_temp_file(temp_output)
        raise

    return LocalExecutionReport(
        status="completed",
        source_path=str(source),
        destination_path=str(destination),
        planned_operations=planned_operations,
        applied_operations=applied_operations,
        filter_chain=filter_chain,
        ffmpeg_command=ffmpeg_command[:-1] + [str(destination)],
        encoding=enc.model_dump(mode="json"),
        resolution_before=resolution_before,
        resolution_after={"width": after_info.width, "height": after_info.height},
        duration_seconds=max(0.0, time.perf_counter() - started),
        warnings=warnings,
        ffprobe_validation=_extract_ffprobe_validation(ffprobe_payload),
    )


def _build_filter_chain(plan: PreprocessingPlan) -> tuple[list[str], list[str], list[str], tuple[int, int] | None]:
    width = plan.source_summary.width
    height = plan.source_summary.height
    if width is None or height is None:
        raise LocalPreprocessingValidationError("Plan source_summary must include width and height for preprocessing.")

    filters: list[str] = []
    applied: list[str] = []
    warnings: list[str] = []
    expected_dims: tuple[int, int] = (width, height)

    for op in plan.operations:
        if not op.enabled:
            continue
        if op.operation_id not in {"crop", "deinterlace", "denoise", "deblock"}:
            raise LocalPreprocessingValidationError(f"Unknown operation_id: {op.operation_id}")
        if op.operation_id in {"denoise", "deblock"}:
            raise LocalPreprocessingValidationError(
                f"Operation '{op.operation_id}' is not supported in this phase. Disable it before execution."
            )
        if op.operation_id == "crop":
            crop_filter, crop_dims, crop_warning = _build_crop_filter(op=op, source_width=width, source_height=height)
            filters.append(crop_filter)
            applied.append("crop")
            expected_dims = crop_dims
            if crop_warning:
                warnings.append(crop_warning)
        elif op.operation_id == "deinterlace":
            filters.append(_build_deinterlace_filter(op=op))
            applied.append("deinterlace")
        else:
            raise LocalPreprocessingValidationError(f"Unsupported operation '{op.operation_id}'.")

    return filters, applied, warnings, expected_dims


def _build_crop_filter(*, op, source_width: int, source_height: int) -> tuple[str, tuple[int, int], str | None]:
    params = op.parameters or {}
    mode = params.get("mode")
    if mode not in {"auto_crop", "custom_crop"}:
        raise LocalPreprocessingValidationError("Enabled crop operation requires mode auto_crop or custom_crop.")
    if params.get("allow_stretch") is not False:
        raise LocalPreprocessingValidationError("Crop operation must keep allow_stretch=false.")
    if params.get("preserve_aspect_ratio") is not True:
        raise LocalPreprocessingValidationError("Crop operation must keep preserve_aspect_ratio=true.")
    rect_payload = params.get("crop_rect")
    if not isinstance(rect_payload, dict):
        raise LocalPreprocessingValidationError("Enabled crop operation requires a crop_rect object.")
    try:
        rect = CropRect.model_validate(rect_payload)
    except Exception as exc:
        raise LocalPreprocessingValidationError(f"Invalid crop_rect payload: {exc}") from exc

    if rect.x + rect.width > source_width or rect.y + rect.height > source_height:
        raise LocalPreprocessingValidationError("Crop rectangle is out of source bounds.")
    if rect.width % 2 != 0 or rect.height % 2 != 0 or rect.x % 2 != 0 or rect.y % 2 != 0:
        raise LocalPreprocessingValidationError(
            "Crop rectangle must use even x/y/width/height for yuv420-compatible processing."
        )
    src_ratio = source_width / source_height
    out_ratio = rect.width / rect.height
    ratio_delta = abs(src_ratio - out_ratio) / src_ratio
    warning = None
    if ratio_delta > 0.03:
        warning = "Crop rectangle changes aspect ratio noticeably; verify visual intent before distant upscale."
    return f"crop={rect.width}:{rect.height}:{rect.x}:{rect.y}", (rect.width, rect.height), warning


def _build_deinterlace_filter(*, op) -> str:
    params = op.parameters or {}
    mode = params.get("mode")
    if mode != "enable":
        raise LocalPreprocessingValidationError("Enabled deinterlace operation requires mode=enable.")
    return "bwdif=mode=send_frame:parity=auto:deint=all"


def _build_ffmpeg_command(
    *,
    source: Path,
    destination: Path,
    filter_chain: str,
    encoding: LocalEncodingOptions,
    ffmpeg_bin: str,
) -> list[str]:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        filter_chain,
        "-c:v",
        encoding.video_codec,
        "-preset",
        encoding.preset,
        "-crf",
        str(encoding.crf),
        "-pix_fmt",
        encoding.pixel_format,
        "-map",
        "0:a?",
        "-map",
        "0:s?",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c:a",
        "copy" if encoding.copy_audio else "aac",
        "-c:s",
        "copy" if encoding.copy_subtitles else "mov_text",
        str(destination),
    ]
    return cmd


def _validate_post_output(*, source_info, output_info, expected_dims: tuple[int, int] | None) -> None:
    if output_info.width is None or output_info.height is None:
        raise LocalPreprocessingExecutionError("Output validation failed: missing output resolution.")
    if expected_dims is not None and (output_info.width, output_info.height) != expected_dims:
        raise LocalPreprocessingExecutionError(
            "Output validation failed: resolution mismatch. "
            f"expected={expected_dims[0]}x{expected_dims[1]} got={output_info.width}x{output_info.height}"
        )
    if source_info.frame_rate_fps and output_info.frame_rate_fps and abs(source_info.frame_rate_fps - output_info.frame_rate_fps) > 0.2:
        raise LocalPreprocessingExecutionError(
            "Output validation failed: frame rate changed unexpectedly. "
            f"source={source_info.frame_rate_fps} output={output_info.frame_rate_fps}"
        )
    if output_info.duration_seconds is not None and output_info.duration_seconds <= 0:
        raise LocalPreprocessingExecutionError("Output validation failed: non-positive output duration.")


def _extract_ffprobe_validation(payload: dict) -> dict:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        streams = []
    video_stream = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
        {},
    )
    audio_count = sum(1 for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio")
    subtitle_count = sum(1 for item in streams if isinstance(item, dict) and item.get("codec_type") == "subtitle")
    return {
        "video_codec": video_stream.get("codec_name"),
        "pixel_format": video_stream.get("pix_fmt"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "audio_stream_count": audio_count,
        "subtitle_stream_count": subtitle_count,
    }


def _build_temp_output_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{destination.stem}.tmp-preproc-",
        suffix=destination.suffix or ".mp4",
        dir=destination.parent,
    )
    os.close(fd)
    path = Path(tmp_path)
    if path.exists():
        path.unlink()
    return path


def _cleanup_temp_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        return
