from __future__ import annotations

import json
from pathlib import Path

from vast_upscaler.offers import OfferCandidate
from vast_upscaler.vast_write_client import InstanceSummary


def enrich_local_report_with_vast_metadata(
    *,
    report_path: Path,
    mode: str,
    requested_worker_image: str | None,
    instance: InstanceSummary | None,
    offer: OfferCandidate | None,
) -> None:
    if not report_path.exists() or not report_path.is_file():
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return

    warnings: list[str] = []
    instance_metadata, hourly_price = _build_instance_metadata(
        mode=mode,
        requested_worker_image=requested_worker_image,
        instance=instance,
        offer=offer,
        warnings=warnings,
    )

    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    processing_seconds = _as_optional_float(timings.get("processing_seconds"))
    total_job_seconds = _as_optional_float(timings.get("total_job_seconds"))
    cost_estimate = _build_cost_estimate(
        hourly_price_usd=hourly_price,
        processing_seconds=processing_seconds,
        total_job_seconds=total_job_seconds,
        mode=mode,
        warnings=warnings,
    )

    existing_instance = payload.get("instance_metadata")
    merged_instance = _deep_merge_dict(existing_instance if isinstance(existing_instance, dict) else {}, instance_metadata)
    existing_cost = payload.get("cost_estimate")
    merged_cost = _deep_merge_dict(existing_cost if isinstance(existing_cost, dict) else {}, cost_estimate)

    payload["instance_metadata"] = merged_instance
    payload["cost_estimate"] = merged_cost

    metric_schema = payload.get("metric_schema") if isinstance(payload.get("metric_schema"), dict) else {}
    provenance_values = metric_schema.get("provenance_values") if isinstance(metric_schema.get("provenance_values"), list) else []
    for token in ("observed", "declared", "estimated", "unavailable"):
        if token not in provenance_values:
            provenance_values.append(token)
    metric_schema["provenance_values"] = provenance_values
    units = metric_schema.get("units") if isinstance(metric_schema.get("units"), dict) else {}
    units.setdefault("cost", "USD")
    units.setdefault("duration", "seconds")
    metric_schema["units"] = units
    field_notes = metric_schema.get("field_notes") if isinstance(metric_schema.get("field_notes"), dict) else {}
    field_notes.setdefault(
        "instance_metadata",
        "Declared by Vast.ai API when available. Sensitive identifiers are excluded by default.",
    )
    field_notes.setdefault(
        "cost_estimate",
        "Estimated from declared hourly price and measured durations. Not guaranteed to match Vast.ai billing.",
    )
    metric_schema["field_notes"] = field_notes
    payload["metric_schema"] = metric_schema

    existing_warnings = payload.get("collection_warnings")
    merged_warnings = list(existing_warnings) if isinstance(existing_warnings, list) else []
    merged_warnings.extend(warnings)
    payload["collection_warnings"] = merged_warnings

    try:
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def _build_instance_metadata(
    *,
    mode: str,
    requested_worker_image: str | None,
    instance: InstanceSummary | None,
    offer: OfferCandidate | None,
    warnings: list[str],
) -> tuple[dict, float | None]:
    hourly_price = None
    price_provenance = "unavailable"
    if instance is not None and instance.hourly_price_usd is not None:
        hourly_price = instance.hourly_price_usd
        price_provenance = "declared"
    elif offer is not None and (offer.dph_total is not None or offer.search_total_hour is not None):
        hourly_price = offer.dph_total if offer.dph_total is not None else offer.search_total_hour
        price_provenance = "declared"
    else:
        warnings.append("vast_hourly_price_unavailable: hourly price missing from instance and offer metadata")

    declared_gpu_model = instance.gpu_name if instance is not None and instance.gpu_name else (offer.gpu_name if offer else None)
    declared_gpu_count = instance.num_gpus if instance is not None and instance.num_gpus is not None else (offer.num_gpus if offer else None)
    declared_location = instance.location if instance is not None and instance.location else (offer.country_or_region if offer else None)
    declared_upload = instance.inet_up_mbps if instance is not None and instance.inet_up_mbps is not None else (offer.inet_up if offer else None)
    declared_download = instance.inet_down_mbps if instance is not None and instance.inet_down_mbps is not None else (offer.inet_down if offer else None)

    if instance is None:
        warnings.append("vast_instance_metadata_partial: instance summary unavailable for report enrichment")

    return (
        {
            "mode": mode,
            "instance_id": instance.instance_id if instance is not None else None,
            "status": instance.actual_status if instance is not None else None,
            "sensitive_identifiers_included": False,
            "provenance": "declared",
            "declared": {
                "gpu": {
                    "model": declared_gpu_model,
                    "count": declared_gpu_count,
                    "vram_mb": instance.gpu_ram_mb if instance is not None else None,
                },
                "cpu": {
                    "model": instance.cpu_name if instance is not None else None,
                    "cores_effective": instance.cpu_cores_effective if instance is not None else None,
                    "threads": instance.cpu_threads if instance is not None else None,
                    "ram_mb": instance.cpu_ram_mb if instance is not None else None,
                },
                "storage": {
                    "disk_gb": instance.disk_space_gb if instance is not None else None,
                },
                "network": {
                    "upload_mbps": declared_upload,
                    "download_mbps": declared_download,
                },
                "location": declared_location,
                "pricing": {
                    "hourly_usd": hourly_price,
                    "provenance": price_provenance,
                },
                "worker_image": {
                    "requested": requested_worker_image,
                    "effective": instance.image_name if instance is not None else None,
                },
            },
        },
        hourly_price,
    )


def _build_cost_estimate(
    *,
    hourly_price_usd: float | None,
    processing_seconds: float | None,
    total_job_seconds: float | None,
    mode: str,
    warnings: list[str],
) -> dict:
    notes = [
        "Estimates are derived from declared hourly price and measured durations.",
        "Estimated amounts may differ from actual Vast.ai billing.",
        "Storage, network, and other provider fees are not included unless explicitly known.",
    ]
    if mode == "manual":
        notes.append("Manual mode can accrue rental cost before and after the measured job window.")

    processing_cost = None
    job_cost = None
    if hourly_price_usd is not None:
        if processing_seconds is not None:
            processing_cost = hourly_price_usd * (processing_seconds / 3600.0)
        else:
            warnings.append("cost_processing_unavailable: timing processing_seconds is unavailable")
        if total_job_seconds is not None:
            job_cost = hourly_price_usd * (total_job_seconds / 3600.0)
        else:
            warnings.append("cost_job_unavailable: timing total_job_seconds is unavailable")
    else:
        warnings.append("cost_estimate_unavailable: hourly price is unavailable")

    return {
        "currency": "USD",
        "hourly_rate_usd": hourly_price_usd,
        "hourly_rate_provenance": "declared" if hourly_price_usd is not None else "unavailable",
        "processing_seconds_used": processing_seconds,
        "total_job_seconds_used": total_job_seconds,
        "processing_cost_estimate_usd": _rounded_or_none(processing_cost),
        "job_cost_estimate_usd": _rounded_or_none(job_cost),
        "provenance": "estimated",
        "notes": notes,
    }


def _deep_merge_dict(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _rounded_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 8)
