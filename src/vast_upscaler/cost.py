"""Financial estimation logic for phase 2 (read-only, hypothesis-driven)."""

from __future__ import annotations

from dataclasses import dataclass

from vast_upscaler.offers import OfferCandidate
from vast_upscaler.video import VideoInfo


@dataclass(frozen=True)
class CostEstimate:
    duration_seconds: float | None
    resolution: str
    estimated_frames: float | None
    selected_gpu: str
    selected_offer_id: int
    rental_total_hourly_usd: float | None
    gpu_hourly_usd: float | None
    storage_hourly_usd: float | None
    transfer_upload_cost_usd: float | None
    transfer_download_cost_usd: float | None
    processing_fps_assumption: float | None
    processing_hours_estimated: float | None
    gpu_cost_estimated_usd: float | None
    storage_cost_estimated_usd: float | None
    total_estimated_usd: float | None
    total_is_partial: bool
    notes: list[str]


def estimate_cost(
    video: VideoInfo,
    offer: OfferCandidate,
    processing_fps_assumption: float | None,
    allocated_storage_gb: int,
) -> CostEstimate:
    duration_seconds = video.duration_seconds
    resolution = _format_resolution(video.width, video.height)
    estimated_frames = _estimate_frame_count(duration_seconds, video.frame_rate_fps)

    rental_total_hourly = offer.search_total_hour or offer.dph_total
    gpu_hourly = offer.search_gpu_hour
    storage_hourly = offer.search_disk_hour

    notes: list[str] = []
    total_is_partial = False

    if processing_fps_assumption is None:
        notes.append("No processing speed provided. Duration and runtime-dependent costs are unknown.")
        total_is_partial = True

    processing_hours_estimated: float | None = None
    if processing_fps_assumption is not None and estimated_frames is not None and processing_fps_assumption > 0:
        processing_hours_estimated = estimated_frames / processing_fps_assumption / 3600.0
        notes.append(
            "Processing duration is a simulation from user-provided fps-processing. "
            "It is not a validated benchmark prediction."
        )
    elif processing_fps_assumption is not None and estimated_frames is None:
        notes.append("Frame count is unknown (missing duration or frame rate), so processing duration remains unknown.")
        total_is_partial = True

    gpu_cost_estimated: float | None = None
    if processing_hours_estimated is not None and rental_total_hourly is not None:
        gpu_cost_estimated = processing_hours_estimated * rental_total_hourly

    storage_cost_estimated: float | None = None
    if processing_hours_estimated is not None:
        if storage_hourly is not None:
            storage_cost_estimated = processing_hours_estimated * storage_hourly
            notes.append("storage_hourly comes from API search.diskHour.")
        elif offer.storage_cost_gb_month is not None:
            monthly = offer.storage_cost_gb_month * float(allocated_storage_gb)
            storage_cost_estimated = monthly * (processing_hours_estimated / (24.0 * 30.0))
            notes.append("Storage cost estimated from storage_cost ($/GB/month); this may not match final billing exactly.")

    total_estimated: float | None = None
    if gpu_cost_estimated is not None:
        # Avoid double-counting storage if rental_total_hourly already includes storage.
        if rental_total_hourly == offer.search_total_hour and offer.search_disk_hour is not None:
            total_estimated = gpu_cost_estimated
            notes.append("search.totalHour appears to include disk cost, so storage is not added twice.")
        elif storage_cost_estimated is not None:
            total_estimated = gpu_cost_estimated + storage_cost_estimated
        else:
            total_estimated = gpu_cost_estimated

    if rental_total_hourly is None:
        total_is_partial = True
        notes.append("Hourly rental price unavailable from API payload.")

    if total_estimated is None:
        total_is_partial = True

    notes.append("Transfer costs are unknown unless explicit API transfer pricing is available for your workflow.")

    return CostEstimate(
        duration_seconds=duration_seconds,
        resolution=resolution,
        estimated_frames=estimated_frames,
        selected_gpu=offer.gpu_name,
        selected_offer_id=offer.offer_id,
        rental_total_hourly_usd=rental_total_hourly,
        gpu_hourly_usd=gpu_hourly,
        storage_hourly_usd=storage_hourly,
        transfer_upload_cost_usd=None,
        transfer_download_cost_usd=None,
        processing_fps_assumption=processing_fps_assumption,
        processing_hours_estimated=processing_hours_estimated,
        gpu_cost_estimated_usd=gpu_cost_estimated,
        storage_cost_estimated_usd=storage_cost_estimated,
        total_estimated_usd=total_estimated,
        total_is_partial=total_is_partial,
        notes=notes,
    )


def _estimate_frame_count(duration_seconds: float | None, frame_rate_fps: float | None) -> float | None:
    if duration_seconds is None or frame_rate_fps is None:
        return None
    return duration_seconds * frame_rate_fps


def _format_resolution(width: int | None, height: int | None) -> str:
    if width is None or height is None:
        return "unknown"
    return f"{width}x{height}"
