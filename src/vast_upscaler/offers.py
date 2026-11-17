"""Offer query, filtering, and ranking logic for Vast.ai phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SortBy = Literal["price", "reliability", "performance"]


@dataclass(frozen=True)
class OfferConstraints:
    gpu_models: list[str]
    max_hourly_price_usd: float
    min_reliability: float
    verified_only: bool
    disk_gb: int
    require_ssh: bool = True
    require_single_gpu: bool = True
    rental_type: str = "ondemand"


@dataclass(frozen=True)
class OfferCandidate:
    offer_id: int
    gpu_name: str
    country_or_region: str
    dph_total: float | None
    reliability: float | None
    cpu_ram_mb: int | None
    disk_space_gb: float | None
    num_gpus: int | None
    direct_port_count: int | None
    verified: bool
    dlperf: float | None
    dlperf_per_dphtotal: float | None
    total_flops: float | None
    inet_down: float | None
    inet_up: float | None
    cpu_cores_effective: int | None
    search_total_hour: float | None
    search_gpu_hour: float | None
    search_disk_hour: float | None
    storage_cost_gb_month: float | None


def build_offers_query(constraints: OfferConstraints, limit: int = 100) -> dict[str, Any]:
    """Builds a strict read-only offer query for POST /api/v0/bundles/."""
    return {
        "type": constraints.rental_type,
        "limit": limit,
        "verified": {"eq": constraints.verified_only},
        "rentable": {"eq": True},
        "gpu_name": {"in": constraints.gpu_models},
        "num_gpus": {"eq": 1} if constraints.require_single_gpu else {"gte": 1},
        "reliability": {"gte": constraints.min_reliability},
        "dph_total": {"lte": constraints.max_hourly_price_usd},
        "disk_space": {"gte": constraints.disk_gb},
        "direct_port_count": {"gte": 1} if constraints.require_ssh else {"gte": 0},
        # Primary sort by total known hourly cost to prioritize financial safety.
        "order": [["dph_total", "asc"]],
    }


def normalize_offers(raw_offers: list[dict[str, Any]]) -> list[OfferCandidate]:
    candidates: list[OfferCandidate] = []
    for raw in raw_offers:
        offer_id = _to_int(raw.get("id"))
        gpu_name = _to_str(raw.get("gpu_name"))
        if offer_id is None or gpu_name is None:
            continue

        search_info = raw.get("search") if isinstance(raw.get("search"), dict) else {}
        country_or_region = _to_str(raw.get("geolocation")) or _to_str(raw.get("geoloc")) or "unknown"

        candidates.append(
            OfferCandidate(
                offer_id=offer_id,
                gpu_name=gpu_name,
                country_or_region=country_or_region,
                dph_total=_to_float(raw.get("dph_total")),
                reliability=_to_float(raw.get("reliability")),
                cpu_ram_mb=_to_int(raw.get("cpu_ram")),
                disk_space_gb=_to_float(raw.get("disk_space")),
                num_gpus=_to_int(raw.get("num_gpus")),
                direct_port_count=_to_int(raw.get("direct_port_count")),
                verified=_is_verified(raw),
                dlperf=_to_float(raw.get("dlperf")),
                dlperf_per_dphtotal=_to_float(raw.get("dlperf_per_dphtotal")),
                total_flops=_to_float(raw.get("total_flops")),
                inet_down=_to_float(raw.get("inet_down")),
                inet_up=_to_float(raw.get("inet_up")),
                cpu_cores_effective=_to_int(raw.get("cpu_cores_effective")),
                search_total_hour=_to_float(search_info.get("totalHour")),
                search_gpu_hour=_to_float(search_info.get("gpuCostPerHour")),
                search_disk_hour=_to_float(search_info.get("diskHour")),
                storage_cost_gb_month=_to_float(raw.get("storage_cost")),
            )
        )
    return candidates


def filter_offers(candidates: list[OfferCandidate], constraints: OfferConstraints) -> list[OfferCandidate]:
    filtered: list[OfferCandidate] = []
    for item in candidates:
        if item.gpu_name not in constraints.gpu_models:
            continue
        if constraints.require_single_gpu and item.num_gpus != 1:
            continue
        if constraints.verified_only and not item.verified:
            continue
        if constraints.require_ssh and (item.direct_port_count or 0) < 1:
            continue
        if item.reliability is None or item.reliability < constraints.min_reliability:
            continue
        if item.disk_space_gb is None or item.disk_space_gb < constraints.disk_gb:
            continue
        if item.dph_total is None or item.dph_total > constraints.max_hourly_price_usd:
            continue
        filtered.append(item)
    return filtered


def rank_offers(candidates: list[OfferCandidate], sort_by: SortBy = "price") -> list[OfferCandidate]:
    if sort_by == "reliability":
        return sorted(
            candidates,
            key=lambda item: (
                -(item.reliability or 0.0),
                item.dph_total if item.dph_total is not None else float("inf"),
                -_indicative_performance(item),
                -_bandwidth_score(item),
                -(item.cpu_cores_effective or 0),
            ),
        )

    if sort_by == "performance":
        return sorted(
            candidates,
            key=lambda item: (
                -_indicative_performance(item),
                item.dph_total if item.dph_total is not None else float("inf"),
                -(item.reliability or 0.0),
                -_bandwidth_score(item),
                -(item.cpu_cores_effective or 0),
            ),
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.dph_total if item.dph_total is not None else float("inf"),
            -(item.reliability or 0.0),
            -_indicative_performance(item),
            -_bandwidth_score(item),
            -(item.cpu_cores_effective or 0),
        ),
    )


def _is_verified(raw: dict[str, Any]) -> bool:
    verified_flag = raw.get("verified")
    if isinstance(verified_flag, bool):
        return verified_flag
    verification_text = _to_str(raw.get("verification"))
    return verification_text == "verified"


def _indicative_performance(item: OfferCandidate) -> float:
    # Provisional ranking signal only; not a direct proxy for video upscaling throughput.
    return item.total_flops or item.dlperf or item.dlperf_per_dphtotal or 0.0


def _bandwidth_score(item: OfferCandidate) -> float:
    return (item.inet_down or 0.0) + (item.inet_up or 0.0)


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
