from vast_upscaler.cost import estimate_cost
from vast_upscaler.offers import OfferCandidate
from vast_upscaler.video import VideoInfo


def _offer() -> OfferCandidate:
    return OfferCandidate(
        offer_id=31129,
        gpu_name="RTX 3090 Ti",
        country_or_region="Korea, KR",
        dph_total=0.228,
        reliability=0.9987,
        cpu_ram_mb=32768,
        disk_space_gb=70,
        num_gpus=1,
        direct_port_count=2,
        verified=True,
        dlperf=25.0,
        dlperf_per_dphtotal=100.0,
        total_flops=40.0,
        inet_down=800,
        inet_up=500,
        cpu_cores_effective=6,
        search_total_hour=0.228,
        search_gpu_hour=0.19,
        search_disk_hour=0.038,
        storage_cost_gb_month=0.12,
    )


def test_estimate_with_processing_speed() -> None:
    video = VideoInfo(file_path="movie.mkv", duration_seconds=600.0, frame_rate_fps=25.0, width=720, height=576)
    estimate = estimate_cost(video=video, offer=_offer(), processing_fps_assumption=5.0, allocated_storage_gb=50)

    assert estimate.estimated_frames == 15000.0
    assert estimate.processing_hours_estimated is not None
    assert estimate.gpu_cost_estimated_usd is not None
    assert estimate.total_estimated_usd is not None
    assert estimate.total_is_partial is False


def test_estimate_without_processing_speed_is_partial() -> None:
    video = VideoInfo(file_path="movie.mkv", duration_seconds=600.0, frame_rate_fps=25.0, width=720, height=576)
    estimate = estimate_cost(video=video, offer=_offer(), processing_fps_assumption=None, allocated_storage_gb=50)

    assert estimate.processing_hours_estimated is None
    assert estimate.total_estimated_usd is None
    assert estimate.total_is_partial is True
