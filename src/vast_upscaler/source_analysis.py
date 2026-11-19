from __future__ import annotations

import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from vast_upscaler.video import VideoProbeError, run_ffprobe


class SourceAnalysisError(RuntimeError):
    pass


class AudioTrackInfo(BaseModel):
    index: int
    codec: str | None = None
    language: str | None = None
    channels: int | None = None
    bitrate_bps: int | None = None


class SubtitleTrackInfo(BaseModel):
    index: int
    codec: str | None = None
    language: str | None = None


class InterlaceInfo(BaseModel):
    field_order: str | None = None
    interlaced: bool | None = None
    confidence: Literal["certain", "uncertain"]
    note: str


class SourceVideoAnalysis(BaseModel):
    file_path: str
    size_bytes: int | None = None
    duration_seconds: float | None = None
    container_format: str | None = None
    overall_bitrate_bps: int | None = None
    video_codec: str | None = None
    video_bitrate_bps: int | None = None
    width: int | None = None
    height: int | None = None
    frame_rate_fps: float | None = None
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    pixel_format: str | None = None
    audio_stream_count: int = 0
    subtitle_stream_count: int = 0
    audio_tracks: list[AudioTrackInfo]
    subtitle_tracks: list[SubtitleTrackInfo]
    interlace: InterlaceInfo


class VisualDiagnosticsOptions(BaseModel):
    sample_count: int = Field(default=5, ge=1, le=20)
    sample_duration_seconds: float = Field(default=1.5, gt=0.1, le=30.0)
    max_analyzed_duration_seconds: float = Field(default=180.0, gt=1.0, le=3600.0)
    ffmpeg_threads: int = Field(default=1, ge=1, le=16)
    ffmpeg_timeout_seconds: float = Field(default=30.0, gt=1.0, le=600.0)
    min_samples_required: int = Field(default=3, ge=1, le=20)


class CropRectangle(BaseModel):
    x: int
    y: int
    width: int
    height: int


class CropdetectDiagnostic(BaseModel):
    status: Literal["ok", "partial", "failed", "skipped"]
    confidence: Literal["high", "medium", "low", "unknown"]
    analyzed_samples: int
    total_samples: int
    probable_margins: dict[str, int] | None = None
    recommended_crop: CropRectangle | None = None
    inconsistent_samples: int = 0
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IdetDiagnostic(BaseModel):
    status: Literal["ok", "partial", "failed", "skipped"]
    confidence: Literal["high", "medium", "low", "unknown"]
    analyzed_samples: int
    total_samples: int
    classification: Literal["progressive", "interlaced", "mixed", "indeterminate"]
    frame_counters: dict[str, int] = Field(default_factory=dict)
    inconsistent_samples: int = 0
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class NoiseArtifactsDiagnostic(BaseModel):
    status: Literal["ok", "partial", "failed", "skipped"]
    confidence: Literal["high", "medium", "low", "unknown"]
    analyzed_samples: int
    total_samples: int
    likely_noise: Literal["low", "medium", "high", "unknown"]
    likely_compression_artifacts: Literal["low", "medium", "high", "unknown"]
    indicators: dict[str, float | int | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class VisualDiagnosticsReport(BaseModel):
    version: str = "1.0"
    sampling_points_seconds: list[float]
    options: VisualDiagnosticsOptions
    cropdetect: CropdetectDiagnostic
    idet: IdetDiagnostic
    noise_artifacts: NoiseArtifactsDiagnostic
    partial: bool
    notes: list[str] = Field(default_factory=list)


def analyze_source_video(video_path: Path, ffprobe_bin: str = "ffprobe") -> SourceVideoAnalysis:
    try:
        payload = run_ffprobe(video_path=video_path, ffprobe_bin=ffprobe_bin)
    except VideoProbeError as exc:
        raise SourceAnalysisError(str(exc)) from exc
    return parse_source_probe(payload=payload, video_path=video_path)


def analyze_source_video_report(
    *,
    video_path: Path,
    include_visual_diagnostics: bool = False,
    diagnostics_options: VisualDiagnosticsOptions | None = None,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, Any]:
    metadata = analyze_source_video(video_path=video_path, ffprobe_bin=ffprobe_bin)
    payload = metadata.model_dump(mode="json")
    if not include_visual_diagnostics:
        return payload

    options = diagnostics_options or VisualDiagnosticsOptions()
    visual = run_visual_diagnostics(
        video_path=video_path,
        metadata=metadata,
        options=options,
        ffmpeg_bin=ffmpeg_bin,
    )
    payload["analysis_report"] = {
        "version": "1.0",
        "metadata_certainty": "Top-level fields come directly from ffprobe metadata parsing when available.",
        "heuristics_certainty": "visual_diagnostics values are heuristic and include explicit confidence levels.",
        "visual_diagnostics": visual.model_dump(mode="json"),
    }
    return payload


def run_visual_diagnostics(
    *,
    video_path: Path,
    metadata: SourceVideoAnalysis,
    options: VisualDiagnosticsOptions,
    ffmpeg_bin: str = "ffmpeg",
) -> VisualDiagnosticsReport:
    samples = _compute_sample_offsets(duration_seconds=metadata.duration_seconds, options=options)
    crop = _run_cropdetect(
        video_path=video_path,
        width=metadata.width,
        height=metadata.height,
        sample_offsets=samples,
        options=options,
        ffmpeg_bin=ffmpeg_bin,
    )
    idet = _run_idet(
        video_path=video_path,
        sample_offsets=samples,
        options=options,
        ffmpeg_bin=ffmpeg_bin,
    )
    noise = _run_noise_artifacts_estimation(
        video_path=video_path,
        metadata=metadata,
        sample_offsets=samples,
        options=options,
        ffmpeg_bin=ffmpeg_bin,
    )
    partial = any(section.status in {"partial", "failed"} for section in (crop, idet, noise))
    notes: list[str] = []
    if metadata.duration_seconds is not None and metadata.duration_seconds < options.sample_duration_seconds:
        notes.append("Video duration is shorter than diagnostic sample window; confidence may be reduced.")
    return VisualDiagnosticsReport(
        sampling_points_seconds=samples,
        options=options,
        cropdetect=crop,
        idet=idet,
        noise_artifacts=noise,
        partial=partial,
        notes=notes,
    )


def parse_source_probe(*, payload: dict, video_path: Path) -> SourceVideoAnalysis:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise SourceAnalysisError("ffprobe payload missing streams list")

    fmt = payload.get("format")
    if not isinstance(fmt, dict):
        fmt = {}

    video_stream = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        raise SourceAnalysisError("No video stream found in input")

    audio_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"]
    subtitle_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "subtitle"]
    interlace = _resolve_interlace_info(video_stream)

    return SourceVideoAnalysis(
        file_path=str(video_path),
        size_bytes=_to_int(fmt.get("size")),
        duration_seconds=_to_float(fmt.get("duration")),
        container_format=_to_str(fmt.get("format_name")),
        overall_bitrate_bps=_to_int(fmt.get("bit_rate")),
        video_codec=_to_str(video_stream.get("codec_name")),
        video_bitrate_bps=_to_int(video_stream.get("bit_rate")),
        width=_to_int(video_stream.get("width")),
        height=_to_int(video_stream.get("height")),
        frame_rate_fps=_parse_rate(video_stream.get("avg_frame_rate")),
        sample_aspect_ratio=_to_str(video_stream.get("sample_aspect_ratio")),
        display_aspect_ratio=_to_str(video_stream.get("display_aspect_ratio")),
        pixel_format=_to_str(video_stream.get("pix_fmt")),
        audio_stream_count=len(audio_streams),
        subtitle_stream_count=len(subtitle_streams),
        audio_tracks=[
            AudioTrackInfo(
                index=_to_int(item.get("index")) or index,
                codec=_to_str(item.get("codec_name")),
                language=_stream_language(item),
                channels=_to_int(item.get("channels")),
                bitrate_bps=_to_int(item.get("bit_rate")),
            )
            for index, item in enumerate(audio_streams)
        ],
        subtitle_tracks=[
            SubtitleTrackInfo(
                index=_to_int(item.get("index")) or index,
                codec=_to_str(item.get("codec_name")),
                language=_stream_language(item),
            )
            for index, item in enumerate(subtitle_streams)
        ],
        interlace=interlace,
    )


def _run_cropdetect(
    *,
    video_path: Path,
    width: int | None,
    height: int | None,
    sample_offsets: list[float],
    options: VisualDiagnosticsOptions,
    ffmpeg_bin: str,
) -> CropdetectDiagnostic:
    if width is None or height is None or width <= 0 or height <= 0:
        return CropdetectDiagnostic(
            status="skipped",
            confidence="unknown",
            analyzed_samples=0,
            total_samples=len(sample_offsets),
            notes=["Source dimensions are unavailable; cropdetect was skipped."],
        )

    pattern = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
    margins: list[dict[str, int]] = []
    errors: list[str] = []
    for offset in sample_offsets:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "info",
            "-ss",
            f"{offset:.3f}",
            "-t",
            f"{options.sample_duration_seconds:.3f}",
            "-threads",
            str(options.ffmpeg_threads),
            "-i",
            str(video_path),
            "-vf",
            "cropdetect=round=2:reset=0",
            "-an",
            "-f",
            "null",
            "-",
        ]
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=options.ffmpeg_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"sample@{offset:.2f}s: {exc}")
            continue
        if run.returncode != 0:
            errors.append(f"sample@{offset:.2f}s: ffmpeg exited with code {run.returncode}")
            continue

        matches = pattern.findall(run.stderr or "")
        if not matches:
            continue
        w, h, x, y = [int(value) for value in matches[-1]]
        left = max(0, x)
        top = max(0, y)
        right = max(0, width - (x + w))
        bottom = max(0, height - (y + h))
        margins.append({"left": left, "top": top, "right": right, "bottom": bottom})

    if not margins:
        status = "failed" if errors else "partial"
        notes = ["No stable crop was detected from sampled segments."]
        if status == "partial":
            notes.append("Likely causes: full-frame content, very dark scenes, or limited sample diversity.")
        return CropdetectDiagnostic(
            status=status,
            confidence="unknown",
            analyzed_samples=0,
            total_samples=len(sample_offsets),
            notes=notes,
            errors=errors,
        )

    medians = {key: round(statistics.median([m[key] for m in margins])) for key in ("left", "top", "right", "bottom")}
    inconsistent = sum(
        1
        for sample in margins
        if any(abs(sample[key] - medians[key]) > 8 for key in ("left", "top", "right", "bottom"))
    )
    rect = _build_crop_rectangle(
        source_width=width,
        source_height=height,
        left=medians["left"],
        top=medians["top"],
        right=medians["right"],
        bottom=medians["bottom"],
    )

    notes: list[str] = []
    if all(value <= 2 for value in medians.values()):
        notes.append("No meaningful black borders were detected.")
    if inconsistent > 0:
        notes.append("Sampled crop results were partially inconsistent across the video timeline.")
    if rect is None:
        notes.append("Computed crop rectangle was invalid after bounds normalization.")

    analyzed = len(margins)
    if analyzed >= options.min_samples_required and inconsistent == 0 and rect is not None:
        confidence = "high"
    elif analyzed >= max(1, options.min_samples_required // 2) and rect is not None:
        confidence = "medium"
    else:
        confidence = "low"

    status = "ok" if analyzed == len(sample_offsets) and not errors else "partial"
    return CropdetectDiagnostic(
        status=status,
        confidence=confidence,
        analyzed_samples=analyzed,
        total_samples=len(sample_offsets),
        probable_margins=medians,
        recommended_crop=rect,
        inconsistent_samples=inconsistent,
        notes=notes,
        errors=errors,
    )


def _run_idet(
    *,
    video_path: Path,
    sample_offsets: list[float],
    options: VisualDiagnosticsOptions,
    ffmpeg_bin: str,
) -> IdetDiagnostic:
    pattern = re.compile(
        r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
    )
    totals = {"tff": 0, "bff": 0, "progressive": 0, "undetermined": 0}
    errors: list[str] = []
    sample_labels: list[str] = []
    analyzed = 0

    for offset in sample_offsets:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "info",
            "-ss",
            f"{offset:.3f}",
            "-t",
            f"{options.sample_duration_seconds:.3f}",
            "-threads",
            str(options.ffmpeg_threads),
            "-i",
            str(video_path),
            "-vf",
            "idet",
            "-an",
            "-f",
            "null",
            "-",
        ]
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=options.ffmpeg_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"sample@{offset:.2f}s: {exc}")
            continue
        if run.returncode != 0:
            errors.append(f"sample@{offset:.2f}s: ffmpeg exited with code {run.returncode}")
            continue

        matches = pattern.findall(run.stderr or "")
        if not matches:
            continue
        tff, bff, progressive, undetermined = [int(value) for value in matches[-1]]
        totals["tff"] += tff
        totals["bff"] += bff
        totals["progressive"] += progressive
        totals["undetermined"] += undetermined
        analyzed += 1
        sample_labels.append(_classify_interlace(tff=tff, bff=bff, progressive=progressive, undetermined=undetermined))

    if analyzed == 0:
        return IdetDiagnostic(
            status="failed" if errors else "partial",
            confidence="unknown",
            analyzed_samples=0,
            total_samples=len(sample_offsets),
            classification="indeterminate",
            frame_counters=totals,
            notes=["No usable idet output was produced."],
            errors=errors,
        )

    classification = _classify_interlace(
        tff=totals["tff"],
        bff=totals["bff"],
        progressive=totals["progressive"],
        undetermined=totals["undetermined"],
    )
    inconsistent = sum(1 for label in sample_labels if label != classification)
    notes: list[str] = []
    if analyzed < options.min_samples_required:
        notes.append("Insufficient sampled segments for a high-confidence interlace diagnosis.")
    if inconsistent > 0:
        notes.append("Interlace detection labels vary across sampled segments.")

    if analyzed < options.min_samples_required or classification == "indeterminate":
        confidence = "low"
    elif analyzed >= options.min_samples_required and inconsistent == 0:
        confidence = "high"
    elif analyzed >= max(1, options.min_samples_required // 2):
        confidence = "medium"
    else:
        confidence = "low"

    status = "ok" if analyzed == len(sample_offsets) and not errors else "partial"
    return IdetDiagnostic(
        status=status,
        confidence=confidence,
        analyzed_samples=analyzed,
        total_samples=len(sample_offsets),
        classification=classification,
        frame_counters=totals,
        inconsistent_samples=inconsistent,
        notes=notes,
        errors=errors,
    )


def _run_noise_artifacts_estimation(
    *,
    video_path: Path,
    metadata: SourceVideoAnalysis,
    sample_offsets: list[float],
    options: VisualDiagnosticsOptions,
    ffmpeg_bin: str,
) -> NoiseArtifactsDiagnostic:
    yavg_pattern = re.compile(r"lavfi\.signalstats\.YAVG=([0-9]+(?:\.[0-9]+)?)")
    ydif_pattern = re.compile(r"lavfi\.signalstats\.YDIF=([0-9]+(?:\.[0-9]+)?)")
    yavg_values: list[float] = []
    ydif_values: list[float] = []
    errors: list[str] = []
    analyzed = 0

    for offset in sample_offsets:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "info",
            "-ss",
            f"{offset:.3f}",
            "-t",
            f"{options.sample_duration_seconds:.3f}",
            "-threads",
            str(options.ffmpeg_threads),
            "-i",
            str(video_path),
            "-vf",
            "signalstats,metadata=print",
            "-an",
            "-f",
            "null",
            "-",
        ]
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=options.ffmpeg_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"sample@{offset:.2f}s: {exc}")
            continue
        if run.returncode != 0:
            errors.append(f"sample@{offset:.2f}s: ffmpeg exited with code {run.returncode}")
            continue

        yavg_values.extend(float(value) for value in yavg_pattern.findall(run.stderr or ""))
        ydif_values.extend(float(value) for value in ydif_pattern.findall(run.stderr or ""))
        analyzed += 1

    indicators: dict[str, float | int | None] = {
        "mean_luma_yavg": _mean_or_none(yavg_values),
        "mean_temporal_luma_diff_ydif": _mean_or_none(ydif_values),
        "luma_stddev": _stddev_or_none(yavg_values),
        "frames_with_signalstats": len(yavg_values),
        "bitrate_per_pixel_per_frame": _bitrate_density(metadata),
    }

    compression = _classify_compression_risk(indicators["bitrate_per_pixel_per_frame"])
    noise = _classify_noise_risk(
        mean_ydif=indicators["mean_temporal_luma_diff_ydif"],
        luma_stddev=indicators["luma_stddev"],
    )
    notes = [
        "Noise/compression estimates are heuristic and may be confounded by motion, fades, and grading choices.",
    ]
    mean_luma = indicators["mean_luma_yavg"]
    if isinstance(mean_luma, float) and mean_luma < 20.0:
        notes.append("Sampled content is very dark; visual diagnostics reliability is reduced.")
    if analyzed < options.min_samples_required:
        notes.append("Insufficient sampled segments for strong confidence.")

    if analyzed == 0:
        status = "failed" if errors else "partial"
        confidence = "unknown"
    elif errors:
        status = "partial"
        confidence = "low"
    else:
        status = "ok"
        confidence = "low"

    if len(yavg_values) == 0:
        notes.append("signalstats did not yield usable frame-level values.")
        if status == "ok":
            status = "partial"

    return NoiseArtifactsDiagnostic(
        status=status,
        confidence=confidence,
        analyzed_samples=analyzed,
        total_samples=len(sample_offsets),
        likely_noise=noise,
        likely_compression_artifacts=compression,
        indicators=indicators,
        notes=notes,
        errors=errors,
    )


def _compute_sample_offsets(*, duration_seconds: float | None, options: VisualDiagnosticsOptions) -> list[float]:
    if duration_seconds is None or duration_seconds <= 0:
        return [0.0]

    effective_duration = min(duration_seconds, options.max_analyzed_duration_seconds)
    window = min(options.sample_duration_seconds, max(0.2, effective_duration))
    if effective_duration <= window:
        return [0.0]

    count = min(options.sample_count, max(1, int(effective_duration // max(0.2, window / 2))))
    if count <= 1:
        return [max(0.0, (effective_duration - window) / 2.0)]

    last_start = max(0.0, effective_duration - window)
    return [round((last_start * i) / (count - 1), 3) for i in range(count)]


def _build_crop_rectangle(
    *,
    source_width: int,
    source_height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> CropRectangle | None:
    x = max(0, left)
    y = max(0, top)
    width = source_width - max(0, left) - max(0, right)
    height = source_height - max(0, top) - max(0, bottom)
    if width <= 0 or height <= 0:
        return None
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    if width <= 0 or height <= 0:
        return None
    return CropRectangle(x=x, y=y, width=width, height=height)


def _classify_interlace(*, tff: int, bff: int, progressive: int, undetermined: int) -> Literal["progressive", "interlaced", "mixed", "indeterminate"]:
    interlaced = tff + bff
    total = interlaced + progressive + undetermined
    if total <= 0:
        return "indeterminate"

    progressive_ratio = progressive / total
    interlaced_ratio = interlaced / total
    if progressive_ratio >= 0.9 and interlaced_ratio <= 0.1:
        return "progressive"
    if interlaced_ratio >= 0.9 and progressive_ratio <= 0.1:
        return "interlaced"
    if progressive_ratio >= 0.2 and interlaced_ratio >= 0.2:
        return "mixed"
    return "indeterminate"


def _classify_compression_risk(value: float | None) -> Literal["low", "medium", "high", "unknown"]:
    if value is None:
        return "unknown"
    if value < 0.035:
        return "high"
    if value < 0.070:
        return "medium"
    return "low"


def _classify_noise_risk(*, mean_ydif: float | None, luma_stddev: float | None) -> Literal["low", "medium", "high", "unknown"]:
    if mean_ydif is None or luma_stddev is None:
        return "unknown"
    if mean_ydif > 15 and luma_stddev < 4:
        return "high"
    if mean_ydif > 10 and luma_stddev < 6:
        return "medium"
    return "unknown"


def _bitrate_density(metadata: SourceVideoAnalysis) -> float | None:
    if (
        metadata.overall_bitrate_bps is None
        or metadata.width is None
        or metadata.height is None
        or metadata.frame_rate_fps is None
        or metadata.width <= 0
        or metadata.height <= 0
        or metadata.frame_rate_fps <= 0
    ):
        return None
    return float(metadata.overall_bitrate_bps) / float(metadata.width * metadata.height * metadata.frame_rate_fps)


def _resolve_interlace_info(video_stream: dict) -> InterlaceInfo:
    field_order = _to_str(video_stream.get("field_order"))
    if field_order is None:
        return InterlaceInfo(
            field_order=None,
            interlaced=None,
            confidence="uncertain",
            note="ffprobe did not provide field_order.",
        )

    normalized = field_order.lower()
    if normalized in {"progressive", "unknown"}:
        return InterlaceInfo(
            field_order=field_order,
            interlaced=False if normalized == "progressive" else None,
            confidence="certain" if normalized == "progressive" else "uncertain",
            note="Interlacing inferred from ffprobe field_order.",
        )

    return InterlaceInfo(
        field_order=field_order,
        interlaced=True,
        confidence="certain",
        note="Interlacing inferred from ffprobe field_order.",
    )


def _parse_rate(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if text in {"", "0/0", "N/A"}:
        return None
    if "/" in text:
        left, right = text.split("/", maxsplit=1)
        num = _to_float(left)
        den = _to_float(right)
        if num is None or den in (None, 0.0):
            return None
        return num / den
    return _to_float(text)


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _stream_language(stream: dict) -> str | None:
    tags = stream.get("tags")
    if not isinstance(tags, dict):
        return None
    value = tags.get("language")
    if isinstance(value, str) and value:
        return value
    return None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _stddev_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.pstdev(values))
