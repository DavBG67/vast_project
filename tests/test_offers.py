from vast_upscaler.offers import (
    OfferConstraints,
    build_offers_query,
    filter_offers,
    normalize_offers,
    rank_offers,
)


def _constraints() -> OfferConstraints:
    return OfferConstraints(
        gpu_models=["RTX 3090 Ti", "RTX 3090", "RTX 4090"],
        max_hourly_price_usd=0.30,
        min_reliability=0.99,
        verified_only=True,
        disk_gb=50,
    )


def test_build_offers_query() -> None:
    query = build_offers_query(_constraints(), limit=42)
    assert query["type"] == "ondemand"
    assert query["gpu_name"]["in"] == ["RTX 3090 Ti", "RTX 3090", "RTX 4090"]
    assert query["direct_port_count"]["gte"] == 1
    assert query["dph_total"]["lte"] == 0.30


def test_filter_by_gpu_reliability_price_disk() -> None:
    raw = [
        {
            "id": 1,
            "gpu_name": "RTX 3090 Ti",
            "num_gpus": 1,
            "dph_total": 0.22,
            "reliability": 0.995,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 60,
        },
        {
            "id": 2,
            "gpu_name": "A100",
            "num_gpus": 1,
            "dph_total": 0.25,
            "reliability": 0.999,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 60,
        },
        {
            "id": 3,
            "gpu_name": "RTX 3090",
            "num_gpus": 1,
            "dph_total": 0.40,
            "reliability": 0.999,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 60,
        },
        {
            "id": 4,
            "gpu_name": "RTX 4090",
            "num_gpus": 2,
            "dph_total": 0.20,
            "reliability": 0.999,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 60,
        },
        {
            "id": 5,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "dph_total": 0.20,
            "reliability": 0.95,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 60,
        },
        {
            "id": 6,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "dph_total": 0.20,
            "reliability": 0.999,
            "verification": "unverified",
            "direct_port_count": 1,
            "disk_space": 60,
        },
        {
            "id": 7,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "dph_total": 0.20,
            "reliability": 0.999,
            "verification": "verified",
            "direct_port_count": 0,
            "disk_space": 60,
        },
        {
            "id": 8,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "dph_total": 0.20,
            "reliability": 0.999,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 10,
        },
    ]

    candidates = normalize_offers(raw)
    compatible = filter_offers(candidates, _constraints())

    assert [item.offer_id for item in compatible] == [1]


def test_rank_offers() -> None:
    raw = [
        {
            "id": 1,
            "gpu_name": "RTX 3090 Ti",
            "num_gpus": 1,
            "dph_total": 0.25,
            "reliability": 0.995,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 60,
            "total_flops": 40,
        },
        {
            "id": 2,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "dph_total": 0.22,
            "reliability": 0.992,
            "verification": "verified",
            "direct_port_count": 1,
            "disk_space": 60,
            "total_flops": 55,
        },
    ]

    candidates = normalize_offers(raw)
    filtered = filter_offers(candidates, _constraints())

    by_price = rank_offers(filtered, sort_by="price")
    by_reliability = rank_offers(filtered, sort_by="reliability")
    by_performance = rank_offers(filtered, sort_by="performance")

    assert by_price[0].offer_id == 2
    assert by_reliability[0].offer_id == 1
    assert by_performance[0].offer_id == 2
