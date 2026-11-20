from __future__ import annotations

import json
from pathlib import Path

from vast_upscaler.offers import OfferCandidate
from vast_upscaler.report_enrichment import enrich_local_report_with_vast_metadata
from vast_upscaler.vast_write_client import InstanceSummary


def _base_report(path: Path) -> None:
    payload = {
        "status": "completed",
        "frames_processed": 100,
        "processing_seconds": 50.0,
        "processing_fps": 2.0,
        "report_schema_version": "2.0",
        "telemetry": {"sampling_interval_seconds": 5.0, "raw_samples": [], "aggregates": {}},
        "transfer_metrics": {
            "upload": {"duration_seconds": 2.0, "bytes_transferred": 1024, "average_mib_per_second": 0.1, "throughput_unit": "MiB/s"},
            "download": {"duration_seconds": 3.0, "bytes_transferred": 2048, "average_mib_per_second": 0.2, "throughput_unit": "MiB/s"},
        },
        "timings": {
            "processing_seconds": 50.0,
            "remote_execution_seconds": 60.0,
            "upload_seconds": 2.0,
            "download_seconds": 3.0,
            "total_job_seconds": 70.0,
        },
        "collection_warnings": [],
        "metric_schema": {"provenance_values": ["observed"], "units": {"duration": "seconds"}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _offer() -> OfferCandidate:
    return OfferCandidate(
        offer_id=1001,
        gpu_name="RTX 4090",
        country_or_region="EU",
        dph_total=0.45,
        reliability=0.99,
        cpu_ram_mb=65536,
        disk_space_gb=500.0,
        num_gpus=1,
        direct_port_count=1,
        verified=True,
        dlperf=10.0,
        dlperf_per_dphtotal=20.0,
        total_flops=100.0,
        inet_down=2000.0,
        inet_up=1000.0,
        cpu_cores_effective=16,
        search_total_hour=0.45,
        search_gpu_hour=0.4,
        search_disk_hour=0.05,
        storage_cost_gb_month=0.2,
    )


def test_manual_instance_metadata_complete(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _base_report(report)
    instance = InstanceSummary(
        instance_id=123,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="1.2.3.4",
        ssh_port=2200,
        gpu_name="RTX 4090",
        num_gpus=1,
        gpu_ram_mb=24576,
        cpu_name="EPYC",
        cpu_cores_effective=16,
        cpu_threads=32,
        cpu_ram_mb=65536,
        disk_space_gb=400.0,
        inet_up_mbps=1200.0,
        inet_down_mbps=2400.0,
        location="DE",
        hourly_price_usd=0.6,
        image_name="ghcr.io/davbg67/vast-video-upscaler-worker:stable",
    )
    enrich_local_report_with_vast_metadata(
        report_path=report,
        mode="manual",
        requested_worker_image="ghcr.io/davbg67/vast-video-upscaler-worker:latest",
        instance=instance,
        offer=None,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["instance_metadata"]["instance_id"] == 123
    assert payload["instance_metadata"]["declared"]["gpu"]["vram_mb"] == 24576
    assert payload["instance_metadata"]["declared"]["worker_image"]["effective"] == "ghcr.io/davbg67/vast-video-upscaler-worker:stable"
    assert payload["instance_metadata"]["sensitive_identifiers_included"] is False
    assert payload["cost_estimate"]["processing_cost_estimate_usd"] is not None
    assert payload["cost_estimate"]["job_cost_estimate_usd"] is not None


def test_manual_instance_partial_metadata(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _base_report(report)
    instance = InstanceSummary(
        instance_id=321,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="1.2.3.4",
        ssh_port=2200,
    )
    enrich_local_report_with_vast_metadata(
        report_path=report,
        mode="manual",
        requested_worker_image=None,
        instance=instance,
        offer=None,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["instance_metadata"]["instance_id"] == 321
    assert payload["cost_estimate"]["hourly_rate_usd"] is None
    assert any("cost_estimate_unavailable" in warning for warning in payload["collection_warnings"])


def test_automatic_uses_offer_when_instance_price_unavailable(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _base_report(report)
    instance = InstanceSummary(
        instance_id=555,
        actual_status="running",
        intended_status="running",
        cur_state="running",
        label="sess",
        image_runtype="ssh_direct",
        ssh_host="1.2.3.4",
        ssh_port=2200,
        gpu_name="RTX 4090",
    )
    enrich_local_report_with_vast_metadata(
        report_path=report,
        mode="automatic",
        requested_worker_image="ghcr.io/davbg67/vast-video-upscaler-worker:tag",
        instance=instance,
        offer=_offer(),
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["instance_metadata"]["declared"]["pricing"]["hourly_usd"] == 0.45
    assert payload["instance_metadata"]["declared"]["network"]["download_mbps"] == 2000.0
    assert payload["cost_estimate"]["hourly_rate_provenance"] == "declared"


def test_api_unavailable_like_missing_instance_data(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _base_report(report)
    enrich_local_report_with_vast_metadata(
        report_path=report,
        mode="automatic",
        requested_worker_image="ghcr.io/davbg67/vast-video-upscaler-worker:tag",
        instance=None,
        offer=_offer(),
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["instance_metadata"]["instance_id"] is None
    assert payload["instance_metadata"]["declared"]["gpu"]["model"] == "RTX 4090"
    assert any("vast_instance_metadata_partial" in warning for warning in payload["collection_warnings"])


def test_price_unknown_leaves_cost_null(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _base_report(report)
    enrich_local_report_with_vast_metadata(
        report_path=report,
        mode="manual",
        requested_worker_image=None,
        instance=None,
        offer=None,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["cost_estimate"]["processing_cost_estimate_usd"] is None
    assert payload["cost_estimate"]["job_cost_estimate_usd"] is None


def test_preserves_existing_v2_sections(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _base_report(report)
    enrich_local_report_with_vast_metadata(
        report_path=report,
        mode="automatic",
        requested_worker_image="ghcr.io/davbg67/vast-video-upscaler-worker:tag",
        instance=None,
        offer=_offer(),
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "telemetry" in payload
    assert "transfer_metrics" in payload
    assert "timings" in payload
    assert payload["metric_schema"]["units"]["duration"] == "seconds"
    assert "unavailable" in payload["metric_schema"]["provenance_values"]
