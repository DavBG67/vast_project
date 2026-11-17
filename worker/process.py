from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
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


_STOP_REQUESTED = False


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
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output_path.with_suffix(output_path.suffix + ".report.json")

    started = time.perf_counter()
    notes: list[str] = []

    try:
        if not input_path.exists():
            raise WorkerError(f"Input file does not exist: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_disk_space(output_path.parent, min_free_gb=args.min_free_gb)

        source = probe_video(input_path)
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
            staged_output = output_path.with_suffix(output_path.suffix + ".tmp")

            frames = run_streaming_upscale(
                input_path=input_path,
                encoded_video_path=encoded_video,
                source_width=source.width,
                source_height=source.height,
                source_fps=source.fps,
                target_width=target_width,
                target_height=target_height,
                upscaler=upscaler,
                crf=args.video_crf,
            )

            mux_with_source_streams(input_path=input_path, video_only_path=encoded_video, output_path=staged_output)
            validate_output(output_path=staged_output, source_duration=source.duration_seconds, source_fps=source.fps)

            if output_path.exists():
                output_path.unlink()
            staged_output.rename(output_path)

            elapsed = time.perf_counter() - started
            fps = (frames / elapsed) if elapsed > 0 else 0.0
            report = ProcessingReport(
                status="completed",
                source=str(input_path),
                output=str(output_path),
                model=model.name,
                profile=args.profile,
                frames_processed=frames,
                processing_seconds=elapsed,
                processing_fps=fps,
                gpu=RealesrganUpscaler.detect_gpu_name(),
                notes=notes,
            )
            write_report(report_path, report)

        print(f"Completed: {output_path}")
        return 0

    except ProfileNotSupportedError as exc:
        return _fail(report_path, args, started, notes + [str(exc)], status="failed")
    except (WorkerError, ProbeError, RealesrganError) as exc:
        return _fail(report_path, args, started, notes + [str(exc)], status="failed")
    except Exception as exc:  # noqa: BLE001
        return _fail(report_path, args, started, notes + [f"Unexpected error: {exc}"], status="failed")


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
) -> int:
    import cv2
    import numpy as np

    decode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-vsync",
        "0",
        "-",
    ]

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

        assert encoder.stdin is not None
        encoder.stdin.close()

        decode_code = decoder.wait()
        encode_code = encoder.wait()

        if decode_code != 0:
            stderr = _safe_decode(decoder.stderr)
            raise WorkerError(f"FFmpeg decode failed: {stderr}")
        if encode_code != 0:
            stderr = _safe_decode(encoder.stderr)
            raise WorkerError(f"FFmpeg encode failed: {stderr}")

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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fail(report_path: Path, args: argparse.Namespace, started: float, notes: list[str], status: str) -> int:
    elapsed = time.perf_counter() - started
    report = ProcessingReport(
        status=status,
        source=str(args.input),
        output=str(args.output),
        model="realesrgan",
        profile=str(args.profile),
        frames_processed=0,
        processing_seconds=elapsed,
        processing_fps=0.0,
        gpu=RealesrganUpscaler.detect_gpu_name(),
        notes=notes,
    )
    write_report(report_path, report)
    for note in notes:
        print(note)
    return 1


def _ensure_disk_space(directory: Path, min_free_gb: float) -> None:
    usage = shutil.disk_usage(directory)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise WorkerError(f"Insufficient disk space in {directory}: {free_gb:.2f} GB available, need {min_free_gb:.2f} GB")


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
